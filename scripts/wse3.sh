#!/usr/bin/env bash
# Scaled config (kx=ky=8, M=N=512): on-chip timing at size.

set -e
# cs_python mounts only the working directory, so always run from the repo root.
cd "$(dirname "$0")/.."

cslc --arch=wse3 csl/layout.csl --fabric-dims=23,10 \
--fabric-offsets=4,1 --params=kernel_x_dim:8,kernel_y_dim:8,M:512,N:512 \
-o out --memcpy --channels 1
cs_python host/run.py --name out
