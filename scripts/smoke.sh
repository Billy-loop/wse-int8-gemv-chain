#!/usr/bin/env bash
# Small config (kx=ky=2, M=N=8): quick correctness check + timing.

set -e
# cs_python mounts only the working directory, so always run from the repo root.
cd "$(dirname "$0")/.."

cslc --arch=wse3 csl/layout.csl --fabric-dims=11,4 \
--fabric-offsets=4,1 --params=kernel_x_dim:2,kernel_y_dim:2,M:8,N:8 \
-o out_smoke --memcpy --channels 1
cs_python host/run.py --name out_smoke
