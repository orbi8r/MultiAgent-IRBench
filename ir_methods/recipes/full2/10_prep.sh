#!/bin/bash
# Prepare data/<run> for every run: events, contextual pieces and the BM25 first stage lists. CPU.
# EST=3000
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
for r in $ALL; do
  D=data/$r
  [ -f $D/.ok_prep ] && continue
  mkdir -p $D
  SECONDS=0
  python $IR/first_stage/lexical_candidates.py $r
  cp $IRBENCH_RUNS/$r/events.jsonl $D/events.jsonl
  echo $SECONDS > $D/lex_secs.txt
  OMP_NUM_THREADS=1 RUNS=$r python $IR/first_stage/write_fuse0.py
  RUNS=$r python $IR/first_stage/reader_pools.py 1
  python $IR/rerankers/gpu_score.py $r z2 Qwen/Qwen3-Embedding-0.6B none 200
  touch $D/.ok_prep
done
python $IR/first_stage/first_stage_timing.py
