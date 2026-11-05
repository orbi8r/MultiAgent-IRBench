#!/bin/bash
# Train ftpub with seed 1 on all public rows and score it over z2. Tag f2_ftpub_s1. GPU.
# EST=6200
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/train_ftpub.py ftpub_s1 1 1
FROZEN_EXTRA='{"train":"all 4964 rows of pub_00 to pub_14, 1 epoch, seed 1, train_reranker.py"}' python $IR/rerankers/score_z2.py z2s1 "f2_ftpub_s1|Qwen/Qwen3-Reranker-0.6B+lora=$WORK/ftpub_s1|build"
rm -rf $WORK/ftpub_s1

