#!/bin/bash
# Listwise 4B instruct reader over the zs06 and ftpub lists. Tags f2_lw4 and f2_lw4pub. GPU.
# EST=3000
# NEEDS=30_z2_base
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/listwise_multi.py $LW4 fp16 "f2_lw4|zs06" "f2_lw4pub|ftpub"
