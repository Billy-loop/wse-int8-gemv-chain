"""Host utilities for symmetric int8 activation quantization."""

import numpy as np

INT8_MAX = 127.0


def decode_symmetric_int8(codes):
  """Decode signed int8 codes to unscaled fp16 integer values."""
  return np.asarray(codes, dtype=np.int8).astype(np.float16)


def quantize_dequantize_symmetric_int8_vector(values):
  """Quantize one vector with one fp16 absmax/127 scale, then dequantize.

  Models the device path: absmax and scale division in fp16, codes from a
  reciprocal-multiply (inv_scale = 1/scale, then values * inv_scale) matching the
  device @fmulh, round-half-away-from-zero (device adds +/-0.5 then truncates
  toward zero), dequant multiply in fp16. Matching this arithmetic keeps the
  device-vs-reference relerr near zero.
  """
  values_arr = np.asarray(values, dtype=np.float16)
  absmax = np.max(np.abs(values_arr))  # fp16; |fp16| is exact
  if absmax == np.float16(0.0):
    scale = np.float16(1.0)
    inv_scale = np.float16(0.0)
  else:
    scale = (absmax / np.float16(INT8_MAX)).astype(np.float16)
    inv_scale = (np.float16(1.0) / scale).astype(np.float16)
  ratio = (values_arr * inv_scale).astype(np.float16)
  adj = np.where(ratio >= np.float16(0.0),
                 (ratio + np.float16(0.5)).astype(np.float16),
                 (ratio - np.float16(0.5)).astype(np.float16))
  codes = np.clip(np.trunc(adj.astype(np.float32)), -127, 127).astype(np.int8)
  dequant = (decode_symmetric_int8(codes) * scale).astype(np.float16)
  return dequant, scale, codes
