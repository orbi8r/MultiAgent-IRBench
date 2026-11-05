#!/bin/bash
# EST=9500
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
# full2 best pipeline (4B ftpub4b2x, fuse0 K200 plus expansion m25) with the per pair score cache kept at $IRBENCH_WORK/rc/ftpub4b2x/<run>.json, timing rows t3_ftpub4b2x, no sub
python $IR/structure_aware/cache_pool_scores_4b.py ftpub4b2x
du -sh $RC/ftpub4b2x | tail -1
