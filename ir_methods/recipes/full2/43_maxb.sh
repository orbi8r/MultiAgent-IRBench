#!/bin/bash
# ftpub over z2 with the piece body capped at 320 and at 640 tokens. Tags f2_maxb320 and f2_maxb640. GPU.
# EST=4500
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
for B in 320 640; do
  MAXB=$B python $IR/rerankers/score_z2.py z2m$B "f2_maxb$B|Qwen/Qwen3-Reranker-0.6B+lora=models/ftpub|build"
done
