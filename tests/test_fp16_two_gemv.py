"""Unit tests for pure-fp16 two-GEMV host helpers."""
import os
import sys

import numpy as np

# The host modules live in host/, next to tests/; put host/ on sys.path so the
# imports resolve regardless of the cwd.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "host"))

from fp16_chain_utils import (
  compute_two_gemv_reference,
  prepare_A_fp16_per_pe,
)


def test_prepare_A_fp16_per_pe_matches_device_column_major_blocks():
  A = np.arange(6 * 8, dtype=np.float16).reshape(6, 8)
  packed, words = prepare_A_fp16_per_pe(A, num_row_blocks=3, num_col_blocks=4)

  assert words == 2
  blocks = packed.view(np.float16).reshape(3, 4, 4)

  # PE row 1, col 2 owns rows [2:4], cols [4:6], stored column-major.
  expected_block = A[2:4, 4:6].T.ravel()
  np.testing.assert_array_equal(blocks[1, 2], expected_block)


def test_prepare_A2_rotated_blocks_for_gemv2_delivery():
  A2 = np.arange(8 * 8, dtype=np.float16).reshape(8, 8)
  kx, ky = 2, 4
  packed_raw, words = prepare_A_fp16_per_pe(A2, num_row_blocks=kx, num_col_blocks=ky)
  packed_for_device = packed_raw.reshape(kx, ky, words).transpose(1, 0, 2).ravel()

  delivered = packed_for_device.view(np.float16).reshape(ky, kx, words * 2)

  # Gemv2 PE(lx=1, pe_y=2) needs rows [4:8], cols [4:6],
  # stored column-major for the local A_dsd walk.
  expected_block = A2[4:8, 4:6].T.ravel()
  np.testing.assert_array_equal(delivered[2, 1], expected_block)


def test_compute_two_gemv_reference_uses_fp16_chain_order():
  A1 = (np.arange(16, dtype=np.float16).reshape(4, 4) / np.float16(10.0))
  A2 = (np.arange(16, dtype=np.float16).reshape(4, 4) / np.float16(20.0))
  x = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float16)

  y1, y2 = compute_two_gemv_reference(A1, A2, x, kx=2, ky=2)

  y1_dense = (A1 @ x).astype(np.float16)
  y2_dense = (A2 @ y1_dense).astype(np.float16)

  assert y1.dtype == np.float16
  assert y2.dtype == np.float16
  np.testing.assert_allclose(y1, y1_dense, rtol=5e-3, atol=5e-3)
  np.testing.assert_allclose(y2, y2_dense, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
  test_prepare_A_fp16_per_pe_matches_device_column_major_blocks()
  test_prepare_A2_rotated_blocks_for_gemv2_delivery()
  test_compute_two_gemv_reference_uses_fp16_chain_order()
  print("ALL TESTS PASSED")
