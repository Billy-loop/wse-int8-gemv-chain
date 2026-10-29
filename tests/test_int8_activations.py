"""Host unit tests for the int8 y1 quantizer and the y1-only quant-aware reference."""
import os
import sys

import numpy as np

# The host modules live in host/, next to tests/; put host/ on sys.path so the
# imports resolve regardless of the cwd.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "host"))

from int8_utils import quantize_dequantize_symmetric_int8_vector
from fp16_chain_utils import (
    compute_two_gemv_reference,
    compute_two_gemv_reference_int8_y1_only,
)


def test_dequant_equals_code_times_scale():
  v = np.array([0.0, 0.3, -0.7, 1.0, -1.0, 0.51], dtype=np.float16)
  dq, scale, codes = quantize_dequantize_symmetric_int8_vector(v)
  assert codes.dtype == np.int8
  assert np.all(np.abs(codes) <= 127)
  # dequant must be exactly code * scale in fp16
  expect = (codes.astype(np.float16) * scale).astype(np.float16)
  np.testing.assert_array_equal(dq, expect)


def test_ones_quantize_losslessly():
  v = np.ones(4, dtype=np.float16)
  dq, scale, codes = quantize_dequantize_symmetric_int8_vector(v)
  # absmax=1 -> code=127, scale=1/127, dequant = 127/127 = 1.0 exactly
  np.testing.assert_array_equal(dq, np.ones(4, dtype=np.float16))
  assert np.all(codes == 127)


def test_all_zero_vector_is_safe():
  v = np.zeros(5, dtype=np.float16)
  dq, scale, codes = quantize_dequantize_symmetric_int8_vector(v)
  assert scale == np.float16(1.0)
  np.testing.assert_array_equal(dq, np.zeros(5, dtype=np.float16))


def test_y1_only_ref_quantizes_only_y1_per_chunk():
  np.random.seed(7)
  N, kx, ky = 8, 2, 2
  A1 = (np.random.rand(N, N) / N).astype(np.float16)
  A2 = (np.random.rand(N, N) / N).astype(np.float16)
  x = np.ones(N, dtype=np.float16)
  y1_q, y2_q = compute_two_gemv_reference_int8_y1_only(A1, A2, x, kx, ky)
  y1_f, _ = compute_two_gemv_reference(A1, A2, x, kx, ky)
  # x is not quantized, so gemv1 reproduces the fp16 y1 exactly; y1 is then
  # quantized once per gemv1-row chunk (ky chunks of N // ky values).
  chunk = N // ky
  for r in range(ky):
    s = slice(r * chunk, (r + 1) * chunk)
    expected, _, _ = quantize_dequantize_symmetric_int8_vector(y1_f[s])
    np.testing.assert_array_equal(y1_q[s], expected)
  assert y2_q.shape == (N,) and y2_q.dtype == np.float16


def test_y1_only_ref_close_to_fp16_when_quant_is_lossless():
  # With kx=ky=N every y1 chunk holds one value, so absmax == |value|,
  # code = +/-127 and code*scale reproduces the value up to fp16 rounding. The
  # hoisted reference computes (A*code)*scale instead of A*(code*scale), so it
  # matches the fp16 reference only up to fp16 rounding of that reordering.
  np.random.seed(1)
  N = 4
  A1 = np.random.rand(N, N).astype(np.float16)
  A2 = np.random.rand(N, N).astype(np.float16)
  x = np.random.rand(N).astype(np.float16)
  y1_q, y2_q = compute_two_gemv_reference_int8_y1_only(A1, A2, x, N, N)
  y1_f, y2_f = compute_two_gemv_reference(A1, A2, x, N, N)
  np.testing.assert_allclose(y1_q, y1_f, rtol=5e-3, atol=5e-3)
  np.testing.assert_allclose(y2_q, y2_f, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
  test_dequant_equals_code_times_scale()
  test_ones_quantize_losslessly()
  test_all_zero_vector_is_safe()
  test_y1_only_ref_quantizes_only_y1_per_chunk()
  test_y1_only_ref_close_to_fp16_when_quant_is_lossless()
  print("int8_utils tests passed")
