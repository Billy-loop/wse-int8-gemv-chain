# Two-GEMV chain with an int8 y1 handoff on Cerebras WSE-3

A small CSL program for the Cerebras WSE-3 that computes a chain of two
matrix-vector products on two tiles of processing elements (PEs):

```text
y1 = A1 @ x
y2 = A2 @ y1
```

The weights `A1`, `A2` and the input `x` are fp16, and all arithmetic is fp16.
The only quantized link is the **handoff of `y1` between the two tiles**: the PE
that finishes a `y1` chunk quantizes it to symmetric int8 (one fp16 absmax/127
scale per chunk) before sending it on, and the second tile accumulates on the
int8 codes and applies the scale once. `y1` never leaves the fabric; only `y2` is
read back. The host checks `y2` against a quantization-aware reference and
reports on-chip kernel cycles measured with the PE timestamp counters.

## Requirements

- The Cerebras SDK, with `cslc` and `cs_python` on `PATH`. Tested with the
  `sdk-cbcore-2.10.0` container (build 202604101435) on the simulator.
- `python3` (standard library only) for `tests/test_packing.py` and
  `tools/parse_breakdown.py`. Everything that needs numpy runs under `cs_python`.

## Quick start

```bash
./scripts/smoke.sh   # kx=ky=2, M=N=8: quick correctness check + timing
./scripts/wse3.sh    # kx=ky=8, M=N=512: timing at a larger size
```

The scripts work from any directory: they `cd` to the repository root, compile
with `cslc`, and run `host/run.py` under `cs_python`. A successful run prints the
total on-chip kernel cycles, the error metrics and `SUCCESS!`.

When running things by hand, start from the repository root: `cs_python` makes
only the current directory visible inside the SDK container.

## Repository layout

```text
csl/
  layout.csl           PE rectangle, per-PE parameters and routes for both tiles
  pe_program.csl       program shared by every PE; tile_kind selects gemv1 or gemv2
host/
  run.py               host runner: prepares data, launches, checks y2, reports timing
  fp16_chain_utils.py  fp16 A-block packing and the reference chains
  int8_utils.py        host model of the symmetric int8 quantizer
  timing_utils.py      reduces per-PE timestamps to total kernel cycles
scripts/
  smoke.sh, wse3.sh    compile + run one configuration each
tools/
  parse_breakdown.py   per-phase cycle breakdown from simulator PROBE prints
tests/                 host unit tests and a simulator packing test
```

## How it works

### PE layout and data flow

The program runs on a `(2*kx) x ky` rectangle of PEs, split into two tiles:

```text
              gemv1: columns [0, kx)               gemv2: columns [kx, 2*kx)
              y1 = A1 @ x                          y2 = A2 @ y1
input         x (fp16), loaded into the top row    y1 (int8) from gemv1, row by row
broadcast     north -> south, fp16, simd_max       west -> east, raw 32-bit words
reduce        west -> east, fp16, simd_max         north -> south, fp16, simd_max
result        y1 chunk on the rightmost column     y2 chunk on the bottom row,
              -> quantized and sent east           read back by the host
```

1. The host copies the fp16 blocks of `A1` (gemv1) and `A2` (gemv2) into PE
   memory, zeroes the `y` buffers, and copies `x` into the top row of gemv1.
2. Each top-row gemv1 PE sends its `x` slice south (two fp16 values per 32-bit
   wavelet with `simd_max`); every gemv1 PE accumulates its partial
   `A1_block @ x_slice` with `@fmach`.
3. The partial sums are added eastward along each row in fp16, so the rightmost
   gemv1 PE of row `r` ends up with chunk `r` of `y1`.
4. That PE quantizes the chunk into an int8 packet and sends it east with
   `@mov32`; the packet reaches every gemv2 PE in row `r`.
5. Each gemv2 PE accumulates `A2_block @ codes` in fp16, multiplies by the
   chunk's scale once, and the partial sums are added southward. The bottom row
   holds `y2`, which the host reads back.

### The int8 y1 packet

For a chunk of `n = N / ky` values the packet has `ceil(n/4) + 1` 32-bit words:

| Word | Content |
| --- | --- |
| 0 | `{ scale: f16, pad: u16 }` |
| 1 .. ceil(n/4) | `{ c0, c1, c2, c3: i8 }`, four signed int8 codes |

The sending PE (`pack_y1_packet` in `csl/pe_program.csl`) computes

- `absmax = max |v|` over the chunk (`@fabsh`, then a halving `@fmaxh` fold),
- `scale = absmax / 127` and `inv_scale = 127 / absmax` (`scale = 1`,
  `inv_scale = 0` when `absmax == 0`),
- `code = clamp(round_half_away_from_zero(v * inv_scale), -127, 127)`,

and packs the whole 4-code words with a single `@map` over lane-strided DSDs,
with a scalar tail for `n % 4`. The receiving PE accumulates
`y += A[:, j] * code_j` and multiplies by `scale` once before the reduce,
instead of dequantizing every value. `host/int8_utils.py` and
`compute_two_gemv_reference_int8_y1_only` in `host/fp16_chain_utils.py` model
this arithmetic and the PE block/reduce order; `host/run.py` checks `y2` against
that reference.

### Data layout

- `A1` is cut into `ky x kx` blocks of `N/ky x N/kx`; gemv1 PE `(column c, row r)`
  holds block `[r][c]`, stored column-major.
- gemv2 is rotated: PE `(local column c, row r)` needs block-row `c` of `A2` and
  the block-column that matches `y1` chunk `r`. The host cuts `A2` into `kx x ky`
  blocks and transposes the block grid before the copy, so that PE receives
  block `[c][r]`.
- fp16 data travels between host and device as pairs packed into 32-bit words
  (`MEMCPY_32BIT`).

### Parameters

`csl/layout.csl` takes `kernel_x_dim` (kx), `kernel_y_dim` (ky), `M` and `N`, and
requires `M == N`, `N % kx == 0` and `N % ky == 0`. `host/run.py` also requires the
PE-local sizes `N/kx` and `N/ky` to be even (fp16 pairs). The scripts use
`--fabric-dims=<2*kx+7>,<ky+2> --fabric-offsets=4,1`, which leaves room around the
rectangle for the memcpy infrastructure.

### Timing

Every PE stamps `t0` at the start of `compute()` and `t1` once its last piece of
work is done (an asynchronous final send triggers the stamp only when it
completes). The host copies all timestamps back and reports

```text
total_kernel_cycles = max over all PEs (t1) - min over all PEs (t0)
time_us             = total_kernel_cycles / 850
```

This excludes host-device copies and simulator wall-clock time. Per-PE spans are
diagnostics only; do not add them up.

## Results

Simulator, `sdk-cbcore-2.10.0`, probes off. Inputs: `A1`, `A2` uniform random
scaled by `1/N`, `x` all ones.

| Configuration | PE rectangle | Total kernel cycles | Time @ 850 MHz | relerr |
| --- | --- | --- | --- | --- |
| `scripts/smoke.sh` (kx=ky=2, M=N=8) | 4 x 2 | 820 | 0.965 us | 1.2e-4 |
| `scripts/wse3.sh` (kx=ky=8, M=N=512) | 16 x 8 | 8078 | 9.504 us | 4.9e-4 |

`relerr = ||y2 - y2_ref||_inf / max(||A2||_inf * ||y1_ref||_inf, 1)` against the
quantization-aware reference; `host/run.py` fails if it reaches `5e-3`.

## Tests

```bash
cs_python tests/test_fp16_two_gemv.py     # fp16 block packing, rotated A2 delivery, fp16 reference
cs_python tests/test_int8_activations.py  # int8 quantizer and the y1-only reference
cs_python tests/test_timing.py            # 48-bit timestamp assembly, total kernel cycles
python3 tests/test_packing.py             # host fp16/u32 packing; simd_max vs simd_off wavelets
```

`tests/test_packing.py` compiles and simulates the kernel twice (kx=ky=4,
M=N=64). It temporarily rewrites the `simd_mode` settings in
`csl/pe_program.csl` and restores them afterwards.

## Per-phase cycle breakdown (optional)

`csl/pe_program.csl` contains simulator print probes, off by default. To see
where the cycles go:

1. set `const ENABLE_PROBE: bool = true;` in `csl/pe_program.csl`,
2. run `./scripts/smoke.sh` (or another configuration) on the simulator,
3. run `python3 tools/parse_breakdown.py sim.log`.

One PE per tile prints the markers. Printing costs cycles, so the tool subtracts
a calibrated per-print cost from each phase. Turn the probes off again before
measuring total kernel cycles.

## Notes

- The simulator writes `sim.log`, `sim_stats.json`, `simconfig.json` (its default
  settings) and `simfab_traces/` into the working directory. `simfab_traces/` is about 2 GB after the 512 run
  and grows with the problem size; delete it when you no longer need it. Build
  and simulator outputs are git-ignored.
- `host/run.py` accepts `--cmaddr <IP:port>` to run on a CS system. The results
  above come from the simulator only.

## Acknowledgements

`csl/layout.csl`, `csl/pe_program.csl` and `host/run.py` started from the
`gemv-08-routes-3` tutorial in
[Cerebras csl-examples](https://github.com/Cerebras/csl-examples)
(Copyright 2025 Cerebras Systems, Apache License 2.0). Those files keep the
original notice and state how they were modified. This is an independent
project, not affiliated with or endorsed by Cerebras Systems.

## License

Apache License 2.0; see [LICENSE](LICENSE).
