#!/bin/bash
# GPU. Key free candidate pools with the ftpub reader (early exit, expansion), the student scores and the family rows.
# Order matters: pool 1, reader, pool 2 from the reader's top 5, reader again, then the submissions.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
set -e
python $IR/first_stage/reader_pools.py 1
for r in $R5; do POOLF=data/$r/pool1.json OUTF=data/$r/pubsub1 python $IR/rerankers/score_pool.py $r models/ftpub 0 1; done
python $IR/rerankers/merge_shards.py pubsub1
python $IR/first_stage/reader_pools.py 2
for r in $R5; do POOLF=data/$r/pool2.json OUTF=data/$r/pubsub2 python $IR/rerankers/score_pool.py $r models/ftpub 0 1; done
python $IR/rerankers/merge_shards.py pubsub2
python $IR/list_builder/write_pool_submissions.py pub                   # tags rec200 early exit rows and the ftpub over fuse0 rows
for s in rstul6 rstul12; do for r in $R5; do python $IR/students/score_student.py $r models/$s $s; done; done
python $IR/list_builder/write_pool_submissions.py stu                   # student alone and student then ftpub cascades
python $IR/first_stage/family_pool.py
for r in $R5; do POOLF=data/$r/fam5_pool.json OUTF=data/$r/fam5_ftpub python $IR/rerankers/score_pool.py $r models/ftpub 0 1; done
python $IR/rerankers/merge_shards.py fam5_ftpub
python $IR/list_builder/write_pool_submissions.py fam ftpub             # fam_* rows
