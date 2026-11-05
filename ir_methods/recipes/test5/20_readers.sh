#!/bin/bash
# GPU. Score the five test runs with every reader of the frozen test study. Output: data/<run>/gpu_<tag>.json.
# The first call builds the z2 first stage (BM25 ctx, BM25 plain and Qwen3 Embedding 0.6B, RRF, top 200), the rest reuse it.
# Adapters and students come from 10_train_public.sh. A reader whose models/ folder is missing is skipped.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
Z=Qwen/Qwen3-Reranker-0.6B; Z4=Qwen/Qwen3-Reranker-4B
try() { [ -e "$1" ] || { echo "no $1, skip"; return 1; }; }
for r in $R5; do
  gs $r z2 Qwen/Qwen3-Embedding-0.6B "$Z,$Z+lora=models/ftpub" 200
  gc $r z4 $Z4
  QINSTR=1 gc $r z3 $Z
  gc $r bgem3 BAAI/bge-reranker-v2-m3
  try models/ftpub4b && gc $r ftpub4b "$Z4+lora=models/ftpub4b"
  try models/ftpub2x && gc $r ftpub2x "$Z+lora=models/ftpub2x"
  try models/ftpub4b2x && gc $r ftpub4b2x "$Z4+lora=models/ftpub4b2x"
  for s in ftsyn_on ftsyn_lit ftsyn_lib fttrans ftreal_lib ftwho_lib; do try models/$s && gc $r $s "$Z+lora=models/$s"; done
  for s in ft4b_on ft4b_lit ft4b_lib; do try models/$s && gc $r $s "$Z4+lora=models/$s"; done
  for s in ce_l6 ce_l12 ce_bgeb ds_bgeb; do try models/$s && MAXB=256 gc $r $s models/$s; done
  for s in ce_l12_512 ce_bgeb512; do try models/$s && MAXB=512 gc $r $s models/$s; done
  [ -e data/$r/.ok_c4b ] || { python $IR/rerankers/cascade.py $r z2 $Z $Z4 50 c4b && touch data/$r/.ok_c4b; }
  [ -e data/$r/list_lw4.json ] || python $IR/rerankers/listwise.py $r z2 $Z Qwen/Qwen3-4B-Instruct-2507 10 lw4
done
