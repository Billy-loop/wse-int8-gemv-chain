"""Pure host-side reduction of per-PE 48-bit TSC timestamps."""
import numpy as np


def make_u48(words):
  """Assemble three little-endian u16 words into one 48-bit integer."""
  return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def summarize_timestamps(start_ts, end_ts, kx, clock_mhz=850.0):
  """Reduce per-PE start/end timestamps to total kernel-cycle metrics."""
  assert start_ts.shape == end_ts.shape, "start/end timestamp shapes differ"
  W, H, words = start_ts.shape
  assert words == 3, f"expected 3 TSC words per PE, got {words}"

  t0 = np.zeros((W, H), dtype=np.int64)
  t1 = np.zeros((W, H), dtype=np.int64)
  for cx in range(W):
    for cy in range(H):
      t0[cx, cy] = make_u48(start_ts[cx, cy])
      t1[cx, cy] = make_u48(end_ts[cx, cy])

  delta = t1 - t0
  total_kernel_cycles = int(t1.max() - t0.min())
  bx, by = np.unravel_index(int(np.argmax(delta)), delta.shape)

  return {
    "total_kernel_cycles": total_kernel_cycles,
    "total_kernel_time_us": total_kernel_cycles / clock_mhz,
    "per_pe_min": int(delta.min()),
    "per_pe_avg": float(delta.mean()),
    "per_pe_max": int(delta.max()),
    "gemv1_max": int(delta[:kx].max()),
    "gemv2_max": int(delta[kx:].max()),
    "bottleneck_pe": (int(bx), int(by)),
  }
