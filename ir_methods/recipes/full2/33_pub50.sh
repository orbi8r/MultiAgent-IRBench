#!/bin/bash
# Train ftpub on every 2nd public row and score it over z2. Tag f2_pub50. GPU.
# EST=4400
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/train_ftpub.py pub50 2 0
FROZEN_EXTRA='{"train":"every 2nd row of pub_00 to pub_14, 1 epoch, seed 0, train_reranker.py"}' python $IR/rerankers/score_z2.py z2pub50 "f2_pub50|Qwen/Qwen3-Reranker-0.6B+lora=$WORK/pub50|build"
rm -rf $WORK/pub50

