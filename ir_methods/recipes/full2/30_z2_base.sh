#!/bin/bash
# The z2 first stage (BM25 ctx, BM25 plain, dense, RRF, top 200) read by the zero shot 0.6B reader and by ftpub. Tags f2_z2_*. GPU.
# EST=3600
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/score_z2.py z2s "f2_z2_first|-|first" "f2_z2_zs06|Qwen/Qwen3-Reranker-0.6B|build" "f2_z2_zs06_raw|Qwen/Qwen3-Reranker-0.6B|raw" "f2_z2_ftpub|Qwen/Qwen3-Reranker-0.6B+lora=models/ftpub|build" "f2_z2_ftpub_raw|Qwen/Qwen3-Reranker-0.6B+lora=models/ftpub|raw"
