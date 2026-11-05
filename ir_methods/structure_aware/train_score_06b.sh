#!/bin/bash
# usage: train_score_06b.sh STYLE SEED SPLITS [freeze]  (called from a recipe after lib.sh): train the 0.6B on the STYLE prefixed rows, then score z2 K200 with the prefix
# adapter goes to $IRBENCH_WORK/adapters3 (about 40 MB), outside the task folder so later tasks find it. Resumes: adapter, subs and gpu markers are skipped when present.
set -e
S=$1; SEED=$2; SPL=$3; NAME=f3_sp06_${S}_s${SEED}${TAGSUF}
STEPCKPT=${STEPCKPT:-1000} python $IR/structure_aware/train_prefix_reader.py $NAME $S $SEED
python $IR/structure_aware/score_prefix_reader.py --tag $NAME --model "Qwen/Qwen3-Reranker-0.6B+lora=$IRBENCH_WORK/adapters3/$NAME" --style $S --splits $SPL ${4:+--freeze} \
  --train "prefixed rows $S of pub_00 to pub_14, 4964 rows, 1 epoch, seed $SEED, train_prefix_reader.py (train_rr recipe)" \
  --section "9.2 full3 trained prefix 0.6B" --method "ftpub recipe trained on $S prefixed public rows over z2 K200, list builder, seed $SEED" \
  --note "prefix $S in training and scoring, control f2_ftpub_s$SEED, style chosen on dev between P1 and P2 for the test sub"
