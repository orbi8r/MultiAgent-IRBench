#!/bin/bash
# Zero shot Qwen3 Reranker 8B over z2 K200 and as a cascade over the zs06 top 50, dev runs. Tags f2_z8b and f2_c8b. GPU.
# EST=8000
# NEEDS=30_z2_base
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/reader_8b.py dev
rm -rf $RC/z8b
