#!/usr/bin/env python3
"""
Verifies the two layers of fp16 packing the kernel relies on:

1. **Host-side u32 packing.** Each pair of consecutive f16s is reinterpreted
   as a single u32 (low 16 bits = first f16, high 16 bits = second) via
   `arr.view(np.uint32)`. The kernel relies on this byte layout being
   identical to what the device sees as `[N]f16`. This test exercises a
   numpy round-trip on a known f16 pattern (executed inside cs_python,
   because the host python lacks numpy).

2. **Fabric-side SIMD packing (the y-reduce).** With
   `.simd_mode = .{ .simd_max = true }` on the fabin/fabout DSDs, two f16s
   share one 32-bit wavelet on the fabric; without it, each f16 ships
   alone. The test builds the kernel twice (simd_max and simd_off), runs
   each through the simulator at a problem size large enough to make the
   y-reduce traffic measurable, and parses `num_wavelets` out of sim.log
   to confirm the simd_max path emits strictly fewer wavelets.

Run from the project root with the host python (the orchestration uses
only stdlib; cs_python is invoked as a subprocess for the numpy bits and
to drive the simulator):

    python3 tests/test_packing.py
"""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# This test lives in tests/; the repo root (csl/, host/) is one level up. All
# subprocess cwd and file paths anchor on REPO.
REPO = Path(__file__).resolve().parent.parent
PE_CSL = REPO / "csl" / "pe_program.csl"
SIM_LOG = REPO / "sim.log"
SIM_STATS = REPO / "sim_stats.json"

# A square two-GEMV smoke problem big enough that fp16 reduce wavelets are
# measurable while still quick in the simulator.
PARAMS = "kernel_x_dim:4,kernel_y_dim:4,M:64,N:64"
KX, KY = 4, 4
M, N = 64, 64
M_PER_PE = M // KY  # gemv1 reduce vector length; gemv2 matches when kx == ky.

SIMD_PATTERN = re.compile(r"\.simd_mode = \.\{ \.simd_(max|off) = true \}")


# ---------------------------------------------------------------------------
# Test 1 — host-side packing round-trip
# ---------------------------------------------------------------------------

HOST_PACKING_CHECK = textwrap.dedent("""\
    import numpy as np

    # f16 -> view(u32) -> view(f16) must be byte-identical.
    rng = np.random.default_rng(7)
    a = rng.standard_normal(64).astype(np.float16)
    packed = a.view(np.uint32)
    assert packed.shape == (32,), f"packed shape wrong: {packed.shape}"

    # Manual unpack: bit-for-bit reconstruction of the two f16s per u32.
    lo = (packed & 0xFFFF).astype(np.uint16).view(np.float16)
    hi = ((packed >> 16) & 0xFFFF).astype(np.uint16).view(np.float16)
    manual = np.empty(64, dtype=np.float16)
    manual[0::2] = lo
    manual[1::2] = hi
    np.testing.assert_array_equal(a, manual)

    # The cheap path used in run.py.
    unpacked = packed.view(np.float16)
    np.testing.assert_array_equal(a, unpacked)

    # Byte-layout invariant: little-endian -> first f16 lives in the
    # low 16 bits of each u32. This is the invariant that makes
    # 2-f16-per-u32 packing valid for memcpy.
    one_pair = np.array([1.0, 2.0], dtype=np.float16)
    u = one_pair.view(np.uint32)[0]
    lo_bits = np.float16(1.0).view(np.uint16)
    hi_bits = np.float16(2.0).view(np.uint16)
    assert (u & 0xFFFF) == lo_bits, \\
        f"low 16 bits should hold first f16, got {u & 0xFFFF:#06x}"
    assert (u >> 16) == hi_bits, \\
        f"high 16 bits should hold second f16, got {u >> 16:#06x}"

    print("OK")
""")


def test_host_packing_roundtrip():
    """`f16 -> view(u32) -> view(f16)` must be byte-identical."""
    result = subprocess.run(
        ["cs_python", "-c", HOST_PACKING_CHECK],
        cwd=REPO, capture_output=True, text=True,
    )
    if result.returncode != 0 or "OK" not in result.stdout:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError("host-side packing round-trip failed inside cs_python")
    print("[1/2] host packing  : OK")


# ---------------------------------------------------------------------------
# Test 2 — fabric SIMD packing (compare wavelet counts)
# ---------------------------------------------------------------------------

def patch_simd_mode(mode: str):
    """Rewrite pe_program.csl in place: every `.simd_mode = .{ .simd_X = true }`
    becomes `.simd_mode = .{ .simd_<mode> = true }`. Returns the previous mode
    so the caller can restore it."""
    source = PE_CSL.read_text()
    match = SIMD_PATTERN.search(source)
    if not match:
        raise RuntimeError(
            "pe_program.csl no longer contains a `.simd_mode = .{ .simd_X = true }` "
            "pattern — was the file restructured? Update this test.")
    prev = match.group(1)
    patched = SIMD_PATTERN.sub(f".simd_mode = .{{ .simd_{mode} = true }}", source)
    PE_CSL.write_text(patched)
    return prev


def build_and_run() -> tuple[int, int]:
    """Compile + run the kernel; returns (num_wavelets, cycle_count)."""
    subprocess.run(
        ["cslc", "--arch=wse3", "csl/layout.csl",
         "--fabric-dims=15,6", "--fabric-offsets=4,1",
         f"--params={PARAMS}",
         "-o", "out", "--memcpy", "--channels", "1"],
        cwd=REPO, check=True, capture_output=True,
    )
    result = subprocess.run(
        ["cs_python", "host/run.py", "--name", "out"],
        cwd=REPO, capture_output=True, text=True,
    )
    if "SUCCESS!" not in result.stdout:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError("kernel did not print SUCCESS!")

    log = SIM_LOG.read_text()
    m = re.search(r"num_wavelets\s+=\s+(\d+)", log)
    if not m:
        raise RuntimeError("num_wavelets not found in sim.log")
    num_wavelets = int(m.group(1))
    cycles = json.loads(SIM_STATS.read_text())["cycle_count"]
    return num_wavelets, cycles


def test_simd_reduces_wavelets():
    original = patch_simd_mode("max")  # snapshot, restore on exit
    try:
        print(f"      problem    : kx={KX}, ky={KY}, M={M}, N={N} → M_per_PE={M_PER_PE}")

        print("      compiling/running with simd_max ...")
        patch_simd_mode("max")
        w_max, c_max = build_and_run()
        print(f"        wavelets={w_max}  cycles={c_max}")

        print("      compiling/running with simd_off ...")
        patch_simd_mode("off")
        w_off, c_off = build_and_run()
        print(f"        wavelets={w_off}  cycles={c_off}")
    finally:
        patch_simd_mode(original)

    saved = w_off - w_max
    pct = 100.0 * saved / w_off
    print(f"      simd_max saves {saved} wavelets ({pct:.2f}% of total)")

    assert saved > 0, (
        f"simd_max should send strictly fewer wavelets than simd_off, "
        f"got {w_max=} {w_off=}.")

    # Each fp16 reduce transition carries M_per_PE f16 values. With simd_off:
    # M_per_PE wavelets; with simd_max: M_per_PE/2 wavelets. The two-GEMV chain
    # has gemv1 EW transitions and gemv2 NS transitions; keep the lower bound
    # conservative so simulator bookkeeping noise does not dominate the test.
    expected_min = (M_PER_PE // 2) * ((KX - 1) * KY + (KY - 1) * KX) // 4
    assert saved >= expected_min, (
        f"savings {saved} is far below expected minimum {expected_min} "
        f"for M_per_PE={M_PER_PE}, kx={KX}, ky={KY}.")

    print("[2/2] fabric SIMD   : OK")


# ---------------------------------------------------------------------------

def main():
    test_host_packing_roundtrip()
    test_simd_reduces_wavelets()
    print("\nall packing tests passed.")


if __name__ == "__main__":
    main()
