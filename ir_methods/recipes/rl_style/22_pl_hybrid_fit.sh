#!/bin/bash
# Refined hybrids h1 to h4 fitted on the dev runs. Tags pl_h*. CPU.
# EST=2400
# NEEDS=21_pl_pool
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
cd "$IRBENCH_REPO" || exit 1
export RL_STYLE_FEATS=$IRBENCH_WORK/rl_style_feats
set -e
python $IR/rl_style/hybrid_fit.py $IRBENCH_OUT
