#!/bin/bash
# Train ftpub on every 4th public row and score it over z2. Tag f2_pub25. GPU.
# EST=3200
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/train_ftpub.py pub25 4 0
FROZEN_EXTRA='{"train":"every 4th row of pub_00 to pub_14, 1 epoch, seed 0, train_reranker.py"}' python $IR/rerankers/score_z2.py z2pub25 "f2_pub25|Qwen/Qwen3-Reranker-0.6B+lora=$WORK/pub25|build"
rm -rf $WORK/pub25

