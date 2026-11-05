#!/bin/bash
# First stage fusion ablations over fuse0 read by ftpub. Tags f2_fl_* and f2_exp_fuse0_*. GPU.
# EST=7200
# NEEDS=20_ftpub_pool
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/fusion.py
rm -rf $RC/ftpub06
