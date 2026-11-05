#!/bin/bash
# GPU, about 11 GB. Train the public trace readers and students. Nothing here reads a benchmark answer key.
# Needs data/<run>/pieces.jsonl for the three dev runs (recipes/00_prep_lexical.sh) because the adapters are only scored on dev.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=Qwen/Qwen3-Reranker-0.6B

# 1. ftpub: public SWE agent trajectories -> pseudo team records -> local Qwen3 4B questions -> rows -> LoRA on the 0.6B reranker.
#    NR pseudo runs of T trajectories each. A second disjoint batch for the 2x adapters: OFFSET=<NR*T> K0=<NR>.
python $IR/training_data/convert_public_traces.py pubruns 16 40
FILES=""
for d in pubruns/pub_*; do
  k=$(basename $d)
  python $IR/training_data/prepare_public_runs.py pieces $d data/$k
  python $IR/training_data/generate_questions.py $k 500 Qwen/Qwen3-4B-Instruct-2507
  [ -s data/$k/synth.jsonl ] || continue
  python $IR/training_data/prepare_public_runs.py rows data/$k
  FILES="$FILES data/$k/train_synth.jsonl"
done
python $IR/rerankers/train_reranker.py $P models/ftpub 1 $FILES

# 2. Larger reader: the same rows on the 4B reranker.
python $IR/rerankers/train_reranker.py Qwen/Qwen3-Reranker-4B models/ftpub4b 1 $FILES

# 3. Cross encoder students trained on the same rows. train_student.py: MiniLM students, MAXLEN env sets the input length (hard labels, 1 epoch).
#    train_cross_encoder.py: base model, output folder, epochs, max length, rows.
#    The exact original invocations of the students were not kept; these are rebuilt from the settings in results/results_test5.csv.
L6=cross-encoder/ms-marco-MiniLM-L-6-v2; L12=cross-encoder/ms-marco-MiniLM-L-12-v2
for pair in "$L6 ce_l6 256" "$L12 ce_l12 256" "$L12 ce_l12_512 512" "$L6 rstul6 512" "$L12 rstul12 512"; do
  set -- $pair; MAXLEN=$3 python $IR/students/train_student.py $1 $2 "data/pub_*/train_synth.jsonl"
done
python $IR/students/train_cross_encoder.py BAAI/bge-reranker-base models/ce_bgeb 1 256 $FILES
python $IR/students/train_cross_encoder.py BAAI/bge-reranker-base models/ce_bgeb512 1 512 $FILES

# 4. Distilled student: teacher scores of the 4B plus half the 0.6B on the public rows, then KL training.
python $IR/students/teacher_scores.py Qwen/Qwen3-Reranker-4B models/ftpub4b teach/t4b_0.jsonl 0 1 16 $FILES
python $IR/students/teacher_scores.py $P models/ftpub teach/t06_0.jsonl 0 1 16 $FILES
python $IR/students/make_teacher_rows.py data/distill.jsonl
python $IR/students/train_cross_encoder.py BAAI/bge-reranker-base models/ds_bgeb 2 512 data/distill.jsonl
