"""Host helpers: fp16 A-block packing and reference two-GEMV chains."""
import numpy as np


def prepare_A_fp16_per_pe(A, num_row_blocks, num_col_blocks):
  """Pack A into PE-local column-major fp16 blocks, viewed as u32 words."""
  assert A.dtype == np.float16, f"expected fp16 A, got {A.dtype}"
  M, N = A.shape
  assert M % num_row_blocks == 0
  assert N % num_col_blocks == 0

  M_per_PE = M // num_row_blocks
  N_per_PE = N // num_col_blocks
  elems_per_pe = M_per_PE * N_per_PE
  assert elems_per_pe % 2 == 0, "fp16 memcpy packing needs even elements per PE"

  prepared = A.reshape(
      num_row_blocks, M_per_PE, num_col_blocks, N_per_PE,
  ).transpose(0, 2, 3, 1).ravel().astype(np.float16)
  return prepared.view(np.uint32), elems_per_pe // 2


def gemv_block_fp16(A_block, x_block):
  """Compute one local block GEMV in fp16 column order."""
  y = np.zeros(A_block.shape[0], dtype=np.float16)
  for col in range(A_block.shape[1]):
    y = (y + (A_block[:, col] * x_block[col]).astype(np.float16)).astype(np.float16)
  return y


def reduce_chain_fp16(partials):
  """Fold PE partials in the same left-to-right/top-to-bottom reduce order."""
  accum = partials[0].astype(np.float16)
  for partial in partials[1:]:
    accum = (partial.astype(np.float16) + accum).astype(np.float16)
  return accum


def compute_two_gemv_reference(A1, A2, x, kx, ky):
  """Reference for y2 = A2 @ (A1 @ x), matching PE block/reduce order."""
  assert A1.dtype == np.float16
  assert A2.dtype == np.float16
  assert x.dtype == np.float16
  assert A1.shape == A2.shape
  N = A1.shape[0]
  assert A1.shape == (N, N)
  assert x.shape == (N,)
  assert N % kx == 0
  assert N % ky == 0

  N_per_PE_1 = N // kx
  M_per_PE_1 = N // ky
  N_per_PE_2 = N // ky
  M_per_PE_2 = N // kx

  y1_chunks = []
  for row_block in range(ky):
    partials = []
    row_start = row_block * M_per_PE_1
    row_end = row_start + M_per_PE_1
    for col_block in range(kx):
      col_start = col_block * N_per_PE_1
      col_end = col_start + N_per_PE_1
      partials.append(gemv_block_fp16(
          A1[row_start:row_end, col_start:col_end],
          x[col_start:col_end],
      ))
    y1_chunks.append(reduce_chain_fp16(partials))
  y1 = np.concatenate(y1_chunks).astype(np.float16)

  y2_chunks = []
  for row_block in range(kx):
    partials = []
    row_start = row_block * M_per_PE_2
    row_end = row_start + M_per_PE_2
    for col_block in range(ky):
      col_start = col_block * N_per_PE_2
      col_end = col_start + N_per_PE_2
      partials.append(gemv_block_fp16(
          A2[row_start:row_end, col_start:col_end],
          y1[col_start:col_end],
      ))
    y2_chunks.append(reduce_chain_fp16(partials))
  y2 = np.concatenate(y2_chunks).astype(np.float16)

  return y1, y2


def compute_two_gemv_reference_int8_y1_only(A1, A2, x, kx, ky):
  """Reference for y2 = A2 @ (A1 @ x) with ONLY y1 int8-quantized.

  The "island" variant: x broadcasts as plain fp16 (Stage 1 is the all-fp16 gemv),
  and only the gemv1->gemv2 y1 handoff is int8-quantized per gemv1-row chunk (ky
  chunks of M_per_PE_1), with the scale hoisted out of the gemv2 loop -- matching
  the device's `@fmach(... code)` loop followed by `@fmulh(y, y, scale)`. The
  reduce stays fp16. Returns (y1_dq, y2): y1_dq is the dequantized y1 fed to gemv2.
  """
  from int8_utils import quantize_dequantize_symmetric_int8_vector
  assert A1.dtype == np.float16 and A2.dtype == np.float16 and x.dtype == np.float16
  assert A1.shape == A2.shape
  N = A1.shape[0]
  assert A1.shape == (N, N) and x.shape == (N,)
  assert N % kx == 0 and N % ky == 0

  N_per_PE_1 = N // kx
  M_per_PE_1 = N // ky
  N_per_PE_2 = N // ky
  M_per_PE_2 = N // kx

  # Stage 1: plain fp16 gemv1 (x is NOT quantized).
  y1_chunks = []
  for row_block in range(ky):
    partials = []
    row_start = row_block * M_per_PE_1
    row_end = row_start + M_per_PE_1
    for col_block in range(kx):
      col_start = col_block * N_per_PE_1
      col_end = col_start + N_per_PE_1
      partials.append(gemv_block_fp16(
          A1[row_start:row_end, col_start:col_end],
          x[col_start:col_end],
      ))
    y1_chunks.append(reduce_chain_fp16(partials))
  y1 = np.concatenate(y1_chunks).astype(np.float16)

  # Stage 2: quantize y1 per gemv1-row chunk; keep codes + scale.
  y1_codes = np.empty_like(y1)
  y1_scales = []
  for r in range(ky):
    s = slice(r * M_per_PE_1, (r + 1) * M_per_PE_1)
    _, scale, codes = quantize_dequantize_symmetric_int8_vector(y1[s])
    y1_codes[s] = codes.astype(np.float16)
    y1_scales.append(scale)

  # gemv2 -> y2: partial = gemv(A2_block, codes) * the chunk's scale, then reduce.
  y2_chunks = []
  for row_block in range(kx):
    partials = []
    row_start = row_block * M_per_PE_2
    row_end = row_start + M_per_PE_2
    for col_block in range(ky):
      col_start = col_block * N_per_PE_2
      col_end = col_start + N_per_PE_2
      p = gemv_block_fp16(
          A2[row_start:row_end, col_start:col_end],
          y1_codes[col_start:col_end],
      )
      p = (p * y1_scales[col_block]).astype(np.float16)
      partials.append(p)
    y2_chunks.append(reduce_chain_fp16(partials))
  y2 = np.concatenate(y2_chunks).astype(np.float16)

  # Dequantized y1 (codes * chunk scale) for the relerr denominator / diagnostics.
  y1_dq = np.empty_like(y1)
  for r in range(ky):
    s = slice(r * M_per_PE_1, (r + 1) * M_per_PE_1)
    y1_dq[s] = (y1_codes[s] * y1_scales[r]).astype(np.float16)

  return y1_dq, y2
