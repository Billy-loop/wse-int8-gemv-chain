# Quantizing inter-operator communication on Cerebras WSE-3

When a computation is laid out spatially on the Cerebras WSE-3, each operator
runs on its own block of processing elements (PEs), and the activation that one
operator produces travels over the on-chip fabric to the next. This repository
asks whether that activation can travel as int8 instead of fp16, and whether
that pays off. The testbed is a chain of two matrix-vector products written in
CSL:

```text
y1 = A1 @ x     gemv1, on one block of PEs
y2 = A2 @ y1    gemv2, on the next block; y1 crosses over as int8
```

**Short answer for this testbed:** int8 cuts the y1 transfer from 63 to 39
cycles (N = 512, 8 x 8 PEs per GEMV), but quantizing y1 costs about 1,300
cycles on the critical path, so the chain runs 17% slower than with fp16. What
decides the outcome is the cost of quantizing on the PE, not the fabric. See
[Results](#results).

## Motivation

The WSE-3 is a grid of PEs. Each PE has its own small SRAM, and PEs exchange
data over a 2D mesh fabric in 32-bit wavelets. The fabric's bandwidth is very
high, but communication is not free. An operator too large for one PE is spread
over a rectangle of PEs: its input is broadcast across the rectangle, its
partial results are reduced, and its output is delivered to the PEs of the next
operator. This traffic grows with the operator, and the premise of this project
is that for large enough operators it stops being negligible next to compute.

Inside an operator, the partial sums of a reduction change at every hop, so
compressing them means re-quantizing on every PE. Between operators, the
payload is an activation vector that is produced once and read by many PEs.
That makes it a natural candidate for compression: quantize it once where it is
produced, send fewer wavelets, and let the consumers compute on the compressed
values.

## The question

Can the activation handed from one operator to the next travel quantized, and
is it worth it? For the chain above, that means sending `y1` from gemv1 to
gemv2 as int8, which raises three questions:

1. **Traffic:** how much shorter does the y1 transfer get?
2. **Compute:** what do quantizing on the sender and dequantizing on the
   receivers cost? Both sit on the critical path.
3. **Accuracy:** how far does `y2` move from the fp16 result?

The measurements below answer the first two for several sizes. The third is
still open.

## The testbed

GEMV is the smallest operator whose input is broadcast and whose output is
reduced across PEs, so two GEMVs in a row are the smallest case with an
activation link between operators. They sit side by side on a `(2*kx) x ky`
rectangle of PEs, one block (called a tile in the code) per GEMV. Drawn with
`kx = ky = 4`; the measurements use 8:

```text
          gemv1 (fp16)                            gemv2 (fp16)
row 0   [ ][ ][ ][Q] --- y1 chunk 0, int8 ---> [ ][ ][ ][ ]
row 1   [ ][ ][ ][Q] --- y1 chunk 1, int8 ---> [ ][ ][ ][ ]
row 2   [ ][ ][ ][Q] --- y1 chunk 2, int8 ---> [ ][ ][ ][ ]
row 3   [ ][ ][ ][Q] --- y1 chunk 3, int8 ---> [ ][ ][ ][ ]
```

- **Only the y1 link is quantized.** The weights, `x`, all arithmetic and both
  reductions stay fp16, so any change in cycles or accuracy comes from the
  link. `y1` never leaves the fabric; only `y2` is read back by the host.
- **Quantized once, read by many.** `Q` marks the PE where a row's y1 chunk
  ends up. It quantizes the chunk once and sends it east, and every gemv2 PE in
  the row receives the same packet. The two blocks are adjacent on the wafer,
  so the link is a single hop followed by a multicast along the row.
- **No re-layout between the operators.** gemv2 is rotated (input broadcast
  west to east, reduction north to south), so the edge where gemv1's output
  lands feeds gemv2's input broadcast directly.

The kernel builds in two versions that differ only in the y1 path. The
compile-time parameter `y1_bits` selects the int8 link (8) or an fp16 baseline
(16) that sends y1 as is, two values per wavelet:

```text
fp16 baseline       gemv1 compute -> reduce y1 -> send y1 (fp16)
(y1_bits = 16)        -> gemv2 compute -> reduce y2
int8 link           gemv1 compute -> reduce y1 -> absmax -> pack -> send y1 (int8)
(y1_bits = 8)         -> gemv2 compute on the int8 codes, scaled once -> reduce y2
```

gemv2 accumulates `A2[:, j] * code_j` directly and multiplies by the chunk's
scale once at the end, instead of turning every code back into an fp16 value
first. An earlier version, not included here, did the latter: it unpacked the
codes into an fp16 buffer and scaled them before the gemv. That extra stage
cost 163–1,373 cycles (N = 256 to 1024), and the chain was 22–30% slower than
fp16 instead of 12–25%.

## Results

Every number below was measured with this repository on the simulator
(`sdk-cbcore-2.10.0`): `scripts/wse3.sh` with `int8` or `fp16` and `M`, `N` set
to each size, 8 x 8 PEs per GEMV (a 16 x 8 rectangle). Each y1 chunk holds
`N/8` values.

### End to end

Total on-chip kernel cycles (see [Timing](#timing)), measured with the probes
off:

| N | fp16 baseline | int8 link |
| ---: | ---: | ---: |
| 128 | 1,801 | 2,256 (+25%) |
| 256 | 3,269 | 3,974 (+22%) |
| 512 | 6,889 | 8,078 (+17%) |
| 1024 | 17,178 | 19,278 (+12%) |

### Where the cycles go

Critical-path stages in cycles, measured on the PEs at the two ends of the
link: the top-right gemv1 PE, which quantizes and sends, and the top-left gemv2
PE, which receives. Stages that are the same in both versions (the x broadcast
and the two reductions) are left out.

| N | Version | gemv1 compute | Quantize | y1 transfer | gemv2 compute |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | fp16 baseline | 1,236 | – | 47 | 1,193 |
| | int8 link | 1,236 | 721 | 31 | 1,185 |
| 512 | fp16 baseline | 2,924 | – | 63 | 2,881 |
| | int8 link | 2,924 | 1,296 | 39 | 2,808 |
| 1024 | fp16 baseline | 7,836 | – | 95 | 7,793 |
| | int8 link | 7,836 | 2,311 | 55 | 7,632 |

- **Quantize:** absmax, scale, quantize and pack, on the sending PE.
- **y1 transfer:** from the moment the sender starts sending y1 until the
  receiver has the whole chunk.
- **gemv2 compute** includes the single scale multiply in the int8 version.

Each stage is the difference between two probe timestamps in `sim.log`, minus a
calibrated cost for the probe prints themselves; see
[Per-phase cycle breakdown](#per-phase-cycle-breakdown-optional).

### Transfer saved vs. quantization spent

| N | y1 values per chunk | Wavelets, fp16 → int8 | Transfer cycles, fp16 → int8 | Saved | Quantize |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 16 → 9 | 47 → 31 | 16 | 721 |
| 512 | 64 | 32 → 17 | 63 → 39 | 24 | 1,296 |
| 1024 | 128 | 64 → 33 | 95 → 55 | 40 | 2,311 |

### What the numbers say

1. **The int8 link does what it is meant to.** The packet carries four codes
   per 32-bit wavelet plus one scale word, `ceil(n/4) + 1` wavelets instead of
   `n/2`, and the transfer is 34–42% shorter.
2. **The transfer was small to begin with.** With fp16 it takes 47–95 cycles,
   0.6–1.4% of the total, and its share falls as N grows: with the grid fixed
   at 8 x 8, the gemv work per PE grows faster than the y1 chunk. Between two
   adjacent operators at these sizes, the fabric is not the bottleneck.
3. **Quantizing costs far more than it saves.** It takes about 20 cycles per y1
   value (18–23), 45–58 times what it saves on the transfer, and it sits on the
   critical path: gemv2 cannot start before the packet arrives.
4. **The overhead is the quantize stage.** gemv2 computing on the int8 codes is
   no slower than on fp16 input, and the end-to-end gap (705, 1,189 and 2,100
   cycles at N = 256, 512 and 1024) is within 10 cycles of the quantize stage
   minus the savings in the transfer and in gemv2.
5. **The overhead falls with size,** from +25% at N = 128 to +12% at
   N = 1024, because the gemv work grows faster than the quantization does.

Side finding: unrolling the column loop in `gemv()` by four cut the gemv
compute stage from 3,322 to 2,924 cycles at N = 512 and from 8,634 to 7,836 at
N = 1024. The stage numbers above include it.

### Correctness

`host/run.py` checks `y2` against a host model of the same build, in the PE
block and reduce order: the int8 arithmetic for the int8 link, the plain fp16
chain for the baseline. It tests the implementation, not the accuracy of int8.
Inputs: `A1`, `A2` uniform random scaled by `1/N`, `x` all ones.

| Configuration | PE rectangle | Total kernel cycles | relerr |
| --- | --- | ---: | ---: |
| `scripts/smoke.sh` (kx=ky=2, M=N=8) | 4 x 2 | 820 | 1.2e-4 |
| `scripts/smoke.sh fp16` | 4 x 2 | 588 | 2.4e-4 |
| `scripts/wse3.sh` (kx=ky=8, M=N=512) | 16 x 8 | 8,078 | 4.9e-4 |
| `scripts/wse3.sh fp16` | 16 x 8 | 6,889 | 4.9e-4 |

`relerr = ||y2 - y2_ref||_inf / max(||A2||_inf * ||y1_ref||_inf, 1)`;
`host/run.py` fails if it reaches `5e-3`. The smoke configuration is only a
correctness check: with four values per chunk, the int8 packet is no shorter
than fp16.

## Open questions

The measurements point at the quantizer, not the fabric. For an int8 link to
pay off, quantizing has to get much cheaper, or the link has to cost much more
than one hop between neighbours.

- **Cheaper quantization.** About 20 cycles per value is the number to beat.
  Options include a scale known in advance instead of a per-chunk absmax, a
  cheaper quantize-and-pack, or another 8-bit format.
- **Quantization off the critical path.** The per-chunk absmax needs the whole
  reduced chunk before quantizing can start. Smaller sub-chunks with their own
  scales, or a scale known in advance, would let quantizing and sending overlap
  with the y1 reduction.
- **Links that cost more.** The premise in [Motivation](#motivation) does not
  show up in these runs: the grid stays at 8 x 8 and the two operators sit next
  to each other, so the link is short and a larger N only adds compute per PE.
  Still to measure: growing the grid with N (a fixed block per PE), operators
  placed far apart, links shared by several transfers, and longer operator
  chains.
- **Accuracy.** How far `y2` moves from an unquantized fp16 evaluation has not
  been characterized.

## Requirements

- The Cerebras SDK, with `cslc` and `cs_python` on `PATH`. Tested with the
  `sdk-cbcore-2.10.0` container (build 202604101435) on the simulator.
- `python3` (standard library only) for `tests/test_packing.py` and
  `tools/parse_breakdown.py`. Everything that needs numpy runs under `cs_python`.

## Quick start

```bash
./scripts/smoke.sh       # kx=ky=2, M=N=8: quick correctness check + timing
./scripts/wse3.sh        # kx=ky=8, M=N=512: timing at a larger size
./scripts/wse3.sh fp16   # the same with the fp16 baseline
```

Both scripts take the y1 handoff format as an optional argument, `int8`
(default) or `fp16`. They work from any directory: they `cd` to the repository
root, compile with `cslc`, and run `host/run.py` under `cs_python`. A successful
run prints the y1 handoff format, the total on-chip kernel cycles, the error
metrics and `SUCCESS!`.

To reproduce the other sizes in [Results](#results), change `M` and `N` in the
`--params` of `scripts/wse3.sh`; the fabric dimensions depend only on `kx` and
`ky`.

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
  smoke.sh, wse3.sh    compile + run one configuration each, int8 or fp16 y1
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

In the fp16 baseline (`y1_bits = 16`), step 4 sends the chunk unquantized, two
fp16 values per wavelet, and step 5 runs the same fp16 loop as gemv1.

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

`csl/layout.csl` takes `kernel_x_dim` (kx), `kernel_y_dim` (ky), `M`, `N` and
`y1_bits` (8 for the int8 link, 16 for the fp16 baseline), and requires
`M == N`, `N % kx == 0` and `N % ky == 0`. `host/run.py` also requires the
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

This is how the stage numbers in [Results](#where-the-cycles-go) were measured.
`csl/pe_program.csl` contains simulator print probes, off by default. To see
where the cycles go:

1. set `const ENABLE_PROBE: bool = true;` in `csl/pe_program.csl`,
2. run `./scripts/smoke.sh` (or another configuration, either build) on the
   simulator,
3. run `python3 tools/parse_breakdown.py sim.log`.

One PE per tile prints the markers. Printing costs cycles, so the tool subtracts
a calibrated per-print cost from each phase. Turn the probes off again before
measuring total kernel cycles.

## Notes

- The simulator writes `sim.log`, `sim_stats.json`, `simconfig.json` (its default
  settings) and `simfab_traces/` into the working directory. `simfab_traces/` is
  about 2 GB after the 512 run and grows with the problem size; delete it when
  you no longer need it. Build and simulator outputs are git-ignored.
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
