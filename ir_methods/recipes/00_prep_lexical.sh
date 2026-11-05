#!/bin/bash
# CPU only. Contextual pieces and the BM25 first stage lists of every run: data/<run>/pieces.jsonl, lex.json, questions.jsonl.
# This is all the baselines in baselines/ need, and the first step of the full first stage in full2/10_prep.sh.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
set -e
for r in $ALL; do python $IR/first_stage/lexical_candidates.py $r; done
