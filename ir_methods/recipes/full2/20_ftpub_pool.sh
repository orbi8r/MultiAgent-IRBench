#!/bin/bash
# Key free candidate pools read by the ftpub reader: pass 1 over the base and fuse0 lists, then the expansion round. GPU.
# EST=5000
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
for r in $ALL; do
  D=data/$r
  [ -f $D/.ok_pool ] && continue
  if [ ! -f $D/pubsub1.json ]; then
    POOLF=$D/pool1.json OUTF=$D/pubsub1 python $IR/rerankers/score_pool.py $r models/ftpub 0 1
    RUNS=$r python $IR/rerankers/merge_shards.py pubsub1
  fi
  [ -f $D/pool2.json ] || RUNS=$r python $IR/first_stage/reader_pools.py 2
  if [ ! -f $D/pubsub2.json ]; then
    POOLF=$D/pool2.json OUTF=$D/pubsub2 python $IR/rerankers/score_pool.py $r models/ftpub 0 1
    RUNS=$r python $IR/rerankers/merge_shards.py pubsub2
  fi
  python $IR/first_stage/merge_reader_pools.py $r
  touch $D/.ok_pool
done
