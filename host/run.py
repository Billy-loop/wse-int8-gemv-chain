#!/usr/bin/env cs_python

# Copyright 2025 Cerebras Systems.
# Modifications copyright 2026 Yu Gao.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from Cerebras csl-examples tutorials/gemv-08-routes-3, modified to
# drive the two-GEMV chain, check y2 against a reference for the chosen y1
# handoff format, and report on-chip kernel cycles.

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
from cerebras.sdk.sdk_utils import memcpy_view

from fp16_chain_utils import (compute_two_gemv_reference,
                              compute_two_gemv_reference_int8_y1_only,
                              prepare_A_fp16_per_pe)
from timing_utils import summarize_timestamps

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Matrix dimensions
N = int(compile_data['params']['N']) # columns
M = int(compile_data['params']['M']) # rows

# PE grid dimensions
kernel_x_dim = int(compile_data['params']['kernel_x_dim'])
kernel_y_dim = int(compile_data['params']['kernel_y_dim'])

# y1 handoff format: 8 = int8 link, 16 = fp16 baseline
y1_bits = int(compile_data['params']['y1_bits'])

np.random.seed(seed=7)

# Scale A by 1/N so each gemv sum stays O(1): y1 ~ N*(0.5/N) = 0.5,
# y2 ~ N*(0.5/N)*0.5 = 0.25 regardless of N. Without this, x=all-ones makes
# y2 ~ N^2/4 which overflows fp16 (65504) at N=512.
A1 = (np.random.rand(N, N) / N).astype(np.float16)
A2 = (np.random.rand(N, N) / N).astype(np.float16)
x  = np.full(N, 1.0, dtype=np.float16)

assert M == N, "two-GEMV chain requires square A"

N_per_PE_1 = N // kernel_x_dim
M_per_PE_1 = N // kernel_y_dim
N_per_PE_2 = N // kernel_y_dim
M_per_PE_2 = N // kernel_x_dim
assert M_per_PE_1 % 2 == 0 and N_per_PE_1 % 2 == 0
assert M_per_PE_2 % 2 == 0 and N_per_PE_2 % 2 == 0

A1_packed, A1_words = prepare_A_fp16_per_pe(
    A1, num_row_blocks=kernel_y_dim, num_col_blocks=kernel_x_dim)

# For gemv2, PE(lx=c, pe_y=r) needs A2 rows from c and columns from r.
# prepare_A_fp16_per_pe(A2, kx, ky) produces block[r][c]; transpose the block
# grid to make ROW_MAJOR memcpy deliver block[c][r] to PE(lx=c, pe_y=r).
A2_packed_raw, A2_words = prepare_A_fp16_per_pe(
    A2, num_row_blocks=kernel_x_dim, num_col_blocks=kernel_y_dim)
A2_packed = A2_packed_raw.reshape(kernel_x_dim, kernel_y_dim, A2_words)\
                         .transpose(1, 0, 2).ravel().astype(np.uint32)

if y1_bits == 8:
  y1_expected, y2_expected = compute_two_gemv_reference_int8_y1_only(
      A1, A2, x, kernel_x_dim, kernel_y_dim)
else:
  y1_expected, y2_expected = compute_two_gemv_reference(
      A1, A2, x, kernel_x_dim, kernel_y_dim)
y2_original = (A2 @ (A1 @ x).astype(np.float16)).astype(np.float16)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
A_symbol = runner.get_id('A')
x_symbol = runner.get_id('x')
y_symbol = runner.get_id('y')
tsc_start_symbol = runner.get_id('tsc_start')
tsc_end_symbol = runner.get_id('tsc_end')

runner.load()
runner.run()

# Initialize y buffers on both tiles to zero. There is no bias term.
y1_init = np.zeros(kernel_x_dim * kernel_y_dim * M_per_PE_1, dtype=np.float16)
y2_init = np.zeros(kernel_x_dim * kernel_y_dim * M_per_PE_2, dtype=np.float16)
runner.memcpy_h2d(y_symbol, y1_init.view(np.uint32),
    0, 0, kernel_x_dim, kernel_y_dim, M_per_PE_1 // 2,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(y_symbol, y2_init.view(np.uint32),
    kernel_x_dim, 0, kernel_x_dim, kernel_y_dim, M_per_PE_2 // 2,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_h2d(A_symbol, A1_packed,
    0, 0, kernel_x_dim, kernel_y_dim, A1_words,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(A_symbol, A2_packed,
    kernel_x_dim, 0, kernel_x_dim, kernel_y_dim, A2_words,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# x -> gemv1 top row only
runner.memcpy_h2d(x_symbol, x.view(np.uint32),
    0, 0, kernel_x_dim, 1, N_per_PE_1 // 2,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.launch('compute', nonblock=False)

# y2 lives on gemv2 bottom row
y2_packed = np.zeros(N // 2, dtype=np.uint32)
runner.memcpy_d2h(y2_packed, y_symbol,
    kernel_x_dim, kernel_y_dim - 1, kernel_x_dim, 1, M_per_PE_2 // 2,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
y2_result = y2_packed.view(np.float16)

# On-chip timing: copy back per-PE start/end timestamps over the full
# (2*kx) x ky rectangle. 3 u16 words per timestamp, MEMCPY_16BIT.
W_pe = 2 * kernel_x_dim
H_pe = kernel_y_dim
TSC_WORDS = 3

tsc_start_u32 = np.zeros(W_pe * H_pe * TSC_WORDS, np.uint32)
runner.memcpy_d2h(tsc_start_u32, tsc_start_symbol,
    0, 0, W_pe, H_pe, TSC_WORDS,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_16BIT, nonblock=False)
tsc_end_u32 = np.zeros(W_pe * H_pe * TSC_WORDS, np.uint32)
runner.memcpy_d2h(tsc_end_u32, tsc_end_symbol,
    0, 0, W_pe, H_pe, TSC_WORDS,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_16BIT, nonblock=False)

# ROW_MAJOR D2H lands axis order (col, row, word).
start_ts = memcpy_view(tsc_start_u32, np.dtype(np.uint16)).reshape(W_pe, H_pe, TSC_WORDS)
end_ts = memcpy_view(tsc_end_u32, np.dtype(np.uint16)).reshape(W_pe, H_pe, TSC_WORDS)

runner.stop()

timing = summarize_timestamps(start_ts, end_ts, kernel_x_dim, clock_mhz=850.0)
assert timing["total_kernel_cycles"] >= timing["per_pe_max"], \
    "total kernel cycles must be at least the slowest single-PE duration"
print(f"y1 handoff: {'int8' if y1_bits == 8 else 'fp16'}")
print("=== total on-chip kernel cycles (excludes H2D/D2H) ===")
print(f"total kernel cycles : {timing['total_kernel_cycles']} cycles "
      f"({timing['total_kernel_time_us']:.3f} us @ 850 MHz)")

print(f"|y2_original - y2_result| = {np.linalg.norm(y2_original - y2_result, np.inf)}")
relerr = np.linalg.norm(y2_result - y2_expected, np.inf) / max(
    np.linalg.norm(A2, np.inf) * np.linalg.norm(y1_expected, np.inf), 1.0)
print(f"relerr = {relerr}")

if N < 10:
  print(f"y2_result   = {y2_result}")
  print(f"y2_expected = {y2_expected}")
  print(f"y1_expected = {y1_expected}")

assert relerr < 5e-3, "y2 wrong"

if N < 10:
  np.testing.assert_allclose(y2_result, y2_expected, rtol=5e-3, atol=5e-3)

print("SUCCESS!")
