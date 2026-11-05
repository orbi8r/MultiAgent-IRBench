#!/bin/bash
# GPU. The whole fast method, timed end to end, tags fast_* and time_fast_*, and the standalone method of tag t1.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
set -e
mkdir -p "$IRBENCH_OUT/fast"
export NPROC=8 OMP_NUM_THREADS=4 STUDENT=models/ds_bgeb
for r in $R5; do
  [ -s $IRBENCH_OUT/fast/fastd_1g_$r.json ] || python $IR/fast_pipeline/timed_run.py $r models/ftpub S100_30,X100 $IRBENCH_OUT/fast/fastd_1g_$r.json
done
python $IR/fast_pipeline/timed_run_subs.py fastd
# t1: lexical K 100 with the public reader, one process per run, writes the submission and its timing dump
mkdir -p "$IRBENCH_OUT/subs/t1"
for r in $R5; do
  python $IR/fast_pipeline/run_method.py $IRBENCH_RUNS/$r $IRBENCH_OUT/subs/t1/$r.json --k 100 --lora models/ftpub --dump $IRBENCH_OUT/fast/t1_$r.dump.json
done
