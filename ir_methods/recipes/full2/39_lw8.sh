#!/bin/bash
# Listwise 8B reader, 4 bit weights, over the zs06 and ftpub lists. Tags f2_lw8 and f2_lw8pub. GPU.
# EST=5400
# NEEDS=30_z2_base
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/listwise_multi.py $LW8 nf4 "f2_lw8|zs06" "f2_lw8pub|ftpub"
