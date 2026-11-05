#!/bin/bash
# The 4B ftpub reader trained on two public batches, over fuse0 top 200, flat and with expansion. Tags f2_ftpub4b2x_*. GPU.
# EST=12000
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/score_pool_4b.py ftpub4b2x
rm -rf $RC/ftpub4b2x
