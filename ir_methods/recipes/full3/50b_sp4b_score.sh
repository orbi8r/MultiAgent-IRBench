#!/bin/bash
# EST=12000
# NEEDS=50a_sp4b_train 30c_sp06_pick
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
# fuse0 K200 flat and with expansion m25 for the 4B trained on prefixed rows: dev subs, then both tags frozen, then test subs once
STYLE=$(cat ${STYLEF:-$IRBENCH_WORK/out/full3_50a_sp4b_train/style.txt} 2>/dev/null || true)
if [ -z "$STYLE" ]; then echo "50a skipped (gate no go)" > $IRBENCH_OUT/SKIPPED.txt; exit 0; fi
python $IR/structure_aware/score_pool_4b_prefix.py $STYLE
rm -rf $RC/f3_sp4b_$STYLE
