#!/bin/bash
# CPU, no neural model. First stage families, their recall, doc2query and the algorithmic readers. Needs data/<run> from 00_prep_lexical.sh.
# Every script that scores against an answer key sets IRBENCH_KEYS itself, the methods never read a key.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
set -e
export OMP_NUM_THREADS=1
python $IR/first_stage/write_fuse0.py                      # fuse0 and its timing, tags rec200_fuse0 and time_fuse0
python $IR/first_stage/arch2_rrf.py                   # arch2_rrf and pass1_rrf lists, tag rec200_arch2_rrf
python $IR/first_stage/cpu_families.py                       # data/<run>/fam_cpu.json, the lexical and structure families
python $IR/first_stage/family_recall.py $IRBENCH_OUT/family_recall.json   # rec200_* rows
python $IR/first_stage/first_stage_recall.py                           # fs_* rows, recall of base and novelty pruned lists at K 10 to 200
export SUBDIR=$IRBENCH_OUT/subs
python $IR/algorithmic/algorithmic_reader.py                  # algread_* grid
python $IR/algorithmic/file_level_reader.py                # filelevel_* variants
python $IR/expansion/conversation_prf.py                    # convprf_* grid
for s in t5 qwen; do for r in $R5; do python $IR/expansion/doc2query_generate.py $r $s; done; done   # GPU optional, a few hours on CPU
python $IR/expansion/doc2query_reader.py t5 qwen                # d2q_* submissions
