#!/usr/bin/env bash
# Scaled config (kx=ky=8, M=N=512): on-chip timing at size.
# Usage: ./scripts/wse3.sh [int8|fp16]   (y1 handoff format, default int8)

set -e
# cs_python mounts only the working directory, so always run from the repo root.
cd "$(dirname "$0")/.."

case "${1:-int8}" in
  int8) y1_bits=8 ;;
  fp16) y1_bits=16 ;;
  *) echo "usage: $0 [int8|fp16]" >&2; exit 1 ;;
esac

cslc --arch=wse3 csl/layout.csl --fabric-dims=23,10 \
--fabric-offsets=4,1 --params=kernel_x_dim:8,kernel_y_dim:8,M:512,N:512,y1_bits:$y1_bits \
-o out --memcpy --channels 1
cs_python host/run.py --name out
