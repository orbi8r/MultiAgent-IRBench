#!/bin/bash
# Create the work folder and fetch the public base models the recipes load. Nothing in the repo holds weights, so run this once.
#   bash ir_methods/setup_work.sh            work folder only
#   bash ir_methods/setup_work.sh models     also download the base models into the Hugging Face cache (tens of GB, the 4B and 8B models are the bulk)
# The work folder is $IRBENCH_WORK (default ir_methods/work, ignored by git). Put it on a disk with room, set IRBENCH_WORK before sourcing.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
echo "work folder: $IRBENCH_WORK"
echo "runs folder: $IRBENCH_RUNS"
ls "$IRBENCH_RUNS" | head -20
if [ "$1" = "models" ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-8B",
          "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B", "BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base",
          "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2", "castorini/doc2query-t5-base-msmarco"]:
    print("fetch", m); snapshot_download(m)
PY
fi
