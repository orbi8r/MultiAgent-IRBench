#!/bin/bash
# Train the original Model A policy with the RL trainer, last epoch, no selection. Tags pl_modelA_*. CPU or GPU.
# EST=6000
# NEEDS=70_pl_feats
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
cd "$IRBENCH_REPO" || exit 1
export RL_STYLE_FEATS=$IRBENCH_WORK/rl_style_feats
set -e
python $IR/rl_style/fit_model_a.py $IRBENCH_OUT
