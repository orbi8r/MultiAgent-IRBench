#!/bin/bash
# Pool features for the refined hybrids, built on our fuse0 plus expansion pool. CPU.
# EST=2400
# NEEDS=20_ftpub_pool 70_pl_feats
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
cd "$IRBENCH_REPO" || exit 1
export RL_STYLE_FEATS=$IRBENCH_WORK/rl_style_feats
set -e
python $IR/rl_style/pool_features.py $IRBENCH_OUT
