#!/bin/bash
# Fit and score the original policies (exact and mh85 settings). Tags pl_dp_*. CPU.
# EST=3000
# NEEDS=70_pl_feats
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
cd "$IRBENCH_REPO" || exit 1
export RL_STYLE_FEATS=$IRBENCH_WORK/rl_style_feats
set -e
python $IR/rl_style/fit_rank.py $IRBENCH_OUT
