#!/bin/bash
# EST=2400
# NEEDS=10_prep_structure
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
# inference only: the existing ftpub reader over z2 K200 with each prefix style, dev runs only, no test sub and no frozen file
for s in P1 P2 P3; do
  python $IR/structure_aware/score_prefix_reader.py --tag f3_probe_$s${TAGSUF} --model "Qwen/Qwen3-Reranker-0.6B+lora=models/ftpub" --style $s --splits dev --kind PROBE \
    --section "9.1 full3 prefix probe" --method "ftpub (trained flat) over z2 K200 with the $s structure prefix on the candidate text, list builder" \
    --note "probe, dev only, how robust the flat reader is to a prefix it never saw; control f2_z2_ftpub"
done
