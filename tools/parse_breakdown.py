#!/usr/bin/env python3
"""Per-phase cycle breakdown from the simprint PROBE markers in sim.log.

Usage:
  1. Set `const ENABLE_PROBE: bool = true;` in csl/pe_program.csl.
  2. Run the simulator, e.g. ./scripts/smoke.sh (sim.log lands in the repo root).
  3. python3 tools/parse_breakdown.py [path/to/sim.log]     (default: ./sim.log)

One representative PE per tile prints the markers (see is_probe() in
csl/pe_program.csl). Printing costs cycles, so each per-tile phase span has an
estimated per-line print cost subtracted, calibrated from back-to-back CAL
markers. Set ENABLE_PROBE back to false before measuring total kernel cycles.
"""
import bisect
import re
import sys
from collections import defaultdict
from statistics import median

# One regex per kind of PROBE line in sim.log. @(\d+) captures the cycle stamp;
# the other groups capture that kind's fields. .*? skips any text between the
# @cycle stamp and "PROBE".
LINE = re.compile(r"@(\d+).*?PROBE t(\d+) e(\d+)")          # phase marker: cycle/tile/event
COL_LINE = re.compile(r"@(\d+).*?PROBE col y(\d+)")         # x-broadcast per-hop probe: cycle/row
MID_LINE = re.compile(r"@(\d+).*?PROBE mid x(\d+) e(\d+)")  # interior-PE recv->exit probe: cycle/col/tag

# Event codes; must match the values passed to probe_mark(event) in csl/pe_program.csl.
START, CAL = 0, 1
GEMV_IN, DEQ_IN, DEQ_OUT, RED_IN = 5, 6, 7, 8
PACKY_IN, PACKY_MID, PACKY_OUT, END = 9, 10, 11, 12
PACKY_fabsh, PACKY_abmax = 15, 16

def parse(path):
    """Scan sim.log line by line and sort the PROBE markers by kind."""
    events = defaultdict(dict)   # events[tile][ev] = cycle of the first occurrence (phase boundary)
    prints = defaultdict(list)   # prints[tile] = cycle of every probe print on that PE (print-cost subtraction)
    cals = defaultdict(list)     # cals[tile]   = all CAL cycles on that tile (calibrates pc)
    col = {}                     # col[row]     = cycle at which x reached that row (gemv() entry)
    mid = {}                     # mid[(col, tag)] = cycle; tag 0 = data received, 1 = exit
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # Try mid -> col -> plain marker in turn; the first match wins (disjoint kinds).
            mm = MID_LINE.search(line)
            if mm:
                cyc, mcol, tag = (int(g) for g in mm.groups())
                mid.setdefault((mcol, tag), cyc)  # keep the first one only
                continue
            mc = COL_LINE.search(line)
            if mc:
                cyc, row = (int(g) for g in mc.groups())
                col.setdefault(row, cyc)
                continue
            m = LINE.search(line)
            if not m:
                continue  # neither a mid/col probe nor a t/e marker -> skip
            cyc, tile, ev = (int(g) for g in m.groups())
            prints[tile].append(cyc)          # record every print (incl. START/CAL/events)
            if ev == CAL:
                cals[tile].append(cyc)         # keep every CAL to measure pc
            elif ev not in events[tile]:
                events[tile][ev] = cyc          # other events: first occurrence only (phase boundary)
    for t in prints:
        prints[t].sort()  # span() bisects over the print stamps, so they must be sorted
    return events, prints, cals, col, mid


def print_cost(cal_list):
    """Estimate the cost of printing one line (pc): median gap between back-to-back CALs."""
    if len(cal_list) < 2:
        return 0.0  # fewer than two CALs: no gap to measure -> 0 (no print-cost subtraction)
    return median(b - a for a, b in zip(cal_list, cal_list[1:]))


def make_span(ev, prints, pc):
    """Return a span(a, b) closure: the real work in phase a->b, print cost removed.

    The number of prints dropped is the count of probe prints stamped in
    [a_stamp, b_stamp) (opening marker counted, closing marker not), found by
    bisecting this PE's print stamps, so it is right whether or not a and b are
    adjacent. Returns None if either marker is missing.
    """
    def span(a, b):
        if a not in ev or b not in ev:
            return None
        a_cyc, b_cyc = ev[a], ev[b]
        drop = bisect.bisect_left(prints, b_cyc) - bisect.bisect_left(prints, a_cyc)
        return (b_cyc - a_cyc) - drop * pc
    return span


def breakdown(tile, ev, prints, cal_list):
    """Print the per-phase cycle breakdown of one tile."""
    name = "gemv1 (y1 = A1 @ x)" if tile == 0 else "gemv2 (y2 = A2 @ y1)"
    pc = print_cost(cal_list)
    span = make_span(ev, prints, pc)

    print(f"\n=== tile {tile}: {name} ===")
    print(f"  per-line print cost ~ {pc:.0f} cyc")
    if not all(e in ev for e in (START, GEMV_IN, RED_IN, END)):
        print("  (incomplete markers; nothing to report)")
        return

    # gemv1's probe PE is a broadcast source (its own send + RAMP loopback);
    # gemv2's receive is mostly waiting for gemv1's y1.
    src = " (broadcast source: incl RAMP loopback)" if tile == 0 else " (incl wait for gemv1)"
    print(f"  receive               : {span(START, GEMV_IN):8.0f} cyc{src}")
    # compute = GEMV_in(5) -> RED_in(8): the @fmach loop + @activate + dispatch
    # into reduce(). On gemv1 the only print inside [5, 8) is GEMV_in itself; on
    # gemv2 with the int8 link there are three (5/6/7, one pc each via bisect),
    # and the interval also covers the int8 accumulate + @fmulh dequant.
    print(f"  compute (gemv->reduce): {span(GEMV_IN, RED_IN):8.0f} cyc")
    # Dequant exists only on gemv2 (@fmulh multiplies the scale back in); it is a
    # sub-item of the compute interval above.
    deq = span(DEQ_IN, DEQ_OUT)
    if deq is not None:
        print(f"    of which dequant    : {deq:8.0f} cyc   (@fmulh, DEQ_in->DEQ_out)")

    if PACKY_IN in ev:  # gemv1 reduce-end PE: sends y1 east (quantizes it first on the int8 link)
        print(f"  E reduce              : {span(RED_IN, PACKY_IN):8.0f} cyc")
        if PACKY_MID in ev:  # int8 link only
            print(f"  quantize y1           : {span(PACKY_IN, PACKY_OUT):8.0f} cyc")
            print(f"    absmax/scale        : {span(PACKY_IN, PACKY_MID):8.0f} cyc")
            print(f"      of which @fabsh   : {span(PACKY_IN, PACKY_fabsh):8.0f} cyc")
            print(f"      of which max fold : {span(PACKY_fabsh, PACKY_abmax):8.0f} cyc")
            print(f"    quant+pack          : {span(PACKY_MID, PACKY_OUT):8.0f} cyc")
        print(f"  send to east          : {span(PACKY_OUT, END):8.0f} cyc")
    else:  # gemv2 reduce-start PE: @fmovh sends the y2 partial south
        print(f"  send y2 (N-S reduce)  : {span(RED_IN, END):8.0f} cyc")
    # The total span is a raw difference of absolute stamps; no print cost removed.
    print(f"  probe PE total span   : {ev[END] - ev[START]} cyc  (START..END, raw)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sim.log"
    events, prints, cals, col, mid = parse(path)
    if not events:  # no markers at all: probes disabled or the simulator was not run
        print(f"No PROBE markers found in {path}. "
              f"Is ENABLE_PROBE=true and was the simulator run?")
        sys.exit(1)

    print(f"Per-phase cycle breakdown from {path}")
    for tile in sorted(events):
        breakdown(tile, events[tile], prints[tile], cals[tile])

    # Cross-PE: propagation of the int8 y1 packet from gemv1 to gemv2.
    # ready = gemv1 PACKY_out (pack-done stamp); recvd = gemv2 GEMV_in (whole packet
    # received). PACKY_out's print runs after its own stamp and before the @mov32
    # send, so it delays the real send by one print; gemv2's GEMV_in is a closing
    # marker and is not subtracted. Hence subtract one gemv1 print cost pc0.
    if (0 in events and 1 in events
            and PACKY_OUT in events[0] and GEMV_IN in events[1]):
        pc0 = print_cost(cals[0])
        ready = events[0][PACKY_OUT]
        recvd = events[1][GEMV_IN]
        prop = recvd - ready - pc0
        print("\n=== inter-PE y1 packet propagation (gemv1 -> gemv2) ===")
        print(f"  gemv1 pack done -> gemv2 received all y1: {prop:8.0f} cyc")
        print(f"    (gemv2.GEMV_in {recvd} - gemv1.PACKY_out {ready} - 1 print {pc0:.0f}); "
              f"int8: ceil(M/4)+1 wavelets, fp16: M/2")

    if 1 in col and 2 in col:
        print("\n=== gemv1 x-broadcast per-hop (NS, rightmost col) ===")
        print(f"  row1 -> row2 (one south hop): {col[2] - col[1]:8.0f} cyc")
        print(f"    (row2 x-arrival {col[2]} - row1 x-arrival {col[1]})")

    cols = sorted({c for (c, _t) in mid})
    if len(cols) >= 2:
        west, east = cols[0], cols[-1]  # larger local_x is further east
        print("\n=== gemv1 interior PEs (row 1, cols {} & {}): recv -> exit ==="
              .format(west, east))
        for c in (east, west):
            if (c, 0) in mid and (c, 1) in mid:
                print(f"  col {c}: recv -> exit = {mid[(c, 1)] - mid[(c, 0)]:8.0f} cyc"
                      f"   (recv {mid[(c, 0)]} -> exit {mid[(c, 1)]})")
        if (east, 1) in mid and (west, 1) in mid:
            print(f"  exit diff (east col {east} - west col {west}): "
                  f"{mid[(east, 1)] - mid[(west, 1)]:8.0f} cyc")

    print("\nNote: one representative PE per tile. Per-tile spans subtract pc per "
          "in-interval probe print; per-hop / interior-PE figures are raw global cycles.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Reference: each reported per-tile figure is span(A, B) = ev[B] - ev[A] minus
# one pc for every probe print stamped in [ev[A], ev[B]) (opening marker counted,
# closing not). Probe-PE print order:
#   tile 0 (gemv1, top-right = broadcast source + reduce end):
#       0,1,1,1, 5, 8, 9, 15, 16, 10, 11, 12   (the gemv1 path emits no 6/7)
#   tile 1 (gemv2, top-left = reduce start):
#       0,1,1,1, 5, 6, 7, 8, 12                (6 = dequant-in, 7 = dequant-out)
#   with the fp16 link (y1_bits = 16) there is nothing to quantize or dequantize:
#       tile 0: 0,1,1,1, 5, 8, 9, 11, 12       tile 1: 0,1,1,1, 5, 8, 12
#
# both tiles:
#   receive             = span(START(0), GEMV_in(5))   [drops START + 3 CAL = 4 pc]
#   compute (gemv->red) = span(GEMV_in(5), RED_in(8))  [loop + @activate + dispatch
#                         into reduce(). gemv1 has only the GEMV_in print in [5,8)
#                         -> drops 1 pc; gemv2 has 5,6,7 -> drops 3 pc, and the
#                         interval also covers the int8 accumulate + @fmulh dequant.]
# gemv2 only:
#   of which dequant    = span(DEQ_in(6), DEQ_out(7))  [drops 1 pc; sub-item of compute]
#   send y2 (N-S)       = span(RED_in(8), END(12))     [drops 1 pc]
# gemv1 only (reduce end):
#   E reduce            = span(RED_in(8), PACKY_in(9))       [drops 1 pc]
#   quantize y1         = span(PACKY_in(9), PACKY_out(11))   [drops 4 pc: 9, 15, 16, 10]
#   quant y1 absmax     = span(PACKY_in(9), PACKY_mid(10))   [drops 3 pc: 9, 15, 16]
#     of which @fabsh   = span(PACKY_in(9), fabsh(15))       [drops 1 pc]
#     of which max fold = span(fabsh(15), abmax(16))         [drops 1 pc]
#   quant y1 pack       = span(PACKY_mid(10), PACKY_out(11)) [drops 1 pc]
#   send to east        = span(PACKY_out(11), END(12))       [drops 1 pc]
#
# cross-PE (global cycles, minus gemv1's PACKY_out print before the @mov32):
#   gemv1 pack done -> gemv2 received all y1
#                       = gemv2.GEMV_in(5) - gemv1.PACKY_out(11) - pc(gemv1)
# per-hop / interior PE: raw global-cycle differences (no pc).
# ---------------------------------------------------------------------------
