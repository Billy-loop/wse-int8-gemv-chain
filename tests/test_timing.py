"""Unit test for pure host-side TSC reduction."""
import os
import sys

import numpy as np

# The host modules live in host/, next to tests/; put host/ on sys.path so the
# imports resolve regardless of the cwd.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "host"))

from timing_utils import make_u48, summarize_timestamps


def _u48_to_words(value):
  return np.array(
    [value & 0xFFFF, (value >> 16) & 0xFFFF, (value >> 32) & 0xFFFF],
    dtype=np.uint16,
  )


def test_make_u48_roundtrip():
  assert make_u48(_u48_to_words(0)) == 0
  assert make_u48(_u48_to_words(1)) == 1
  assert make_u48(_u48_to_words(0x1_0000)) == 0x1_0000
  assert make_u48(_u48_to_words(0xABCD_1234_5678)) == 0xABCD_1234_5678


def test_summarize_timestamps_reports_total_kernel_cycles():
  kx, W, H = 2, 4, 2
  start = np.zeros((W, H, 3), dtype=np.uint16)
  end = np.zeros((W, H, 3), dtype=np.uint16)

  starts = {
    (0, 0): 100, (0, 1): 98,
    (1, 0): 102, (1, 1): 101,
    (2, 0): 105, (2, 1): 97,
    (3, 0): 99,  (3, 1): 104,
  }
  durations = {
    (0, 0): 50, (0, 1): 60,
    (1, 0): 55, (1, 1): 65,
    (2, 0): 120, (2, 1): 130,
    (3, 0): 140, (3, 1): 200,
  }
  for coord, dur in durations.items():
    t0 = starts[coord]
    start[coord] = _u48_to_words(t0)
    end[coord] = _u48_to_words(t0 + dur)

  metrics = summarize_timestamps(start, end, kx, clock_mhz=850.0)

  assert metrics["total_kernel_cycles"] == 207
  assert abs(metrics["total_kernel_time_us"] - 207 / 850.0) < 1e-9
  assert metrics["total_kernel_cycles"] >= metrics["per_pe_max"]
  assert metrics["per_pe_min"] == 50
  assert metrics["per_pe_max"] == 200
  assert metrics["gemv1_max"] == 65
  assert metrics["gemv2_max"] == 200
  assert metrics["bottleneck_pe"] == (3, 1)


if __name__ == "__main__":
  test_make_u48_roundtrip()
  test_summarize_timestamps_reports_total_kernel_cycles()
  print("ALL TESTS PASSED")
