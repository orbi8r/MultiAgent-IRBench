#!/bin/bash
# Features of the original RL style method on the eight runs. Run from the repo root, it imports method/. CPU.
# EST=3600
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
cd "$IRBENCH_REPO" || exit 1
export RL_STYLE_FEATS=$IRBENCH_WORK/rl_style_feats
set -e
python $IR/rl_style/features.py $IRBENCH_OUT
