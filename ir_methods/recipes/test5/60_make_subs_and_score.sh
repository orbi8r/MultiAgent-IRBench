#!/bin/bash
# CPU. Frozen submissions from the saved reader outputs, then the scores. Each row is skipped when its gpu files are missing.
# write_submissions.py runs the frozen list builder (EARLIEST=1 LOC_SCAN=30 UTIL_A=0). The scorer is the repo score.py through aggregate/score_test_subs.py.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
done_() { for r in $R5; do [ -s $IRBENCH_OUT/subs/$1/$r.json ] || return 1; done; }
has() { for r in $R5; do [ -s data/$r/gpu_$1.json ] || return 1; done; }
# row NAME NEEDTAG... then the separator -- then the write_submissions.py arguments
row() { local name=$1; shift; local need=(); while [ "$1" != "--" ]; do need+=($1); shift; done; shift
  done_ $name && { echo skip $name; return; }
  for t in "${need[@]}"; do has $t || { echo "missing gpu_$t for $name"; return; }; done
  python $IR/list_builder/write_submissions.py "$@" || echo "FAILED $name"; }
Z=Qwen/Qwen3-Reranker-0.6B; FP="$Z+lora=models/ftpub"; Z4=Qwen/Qwen3-Reranker-4B; F4="$Z4+lora=models/ftpub4b"
row zs06 z2 -- gpu zs06 z2 $Z
row ftpub z2 -- gpu ftpub z2 "$FP"
row zs06_nolb z2 -- raw zs06_nolb z2 $Z
row z2_first z2 -- first z2_first z2 200
row ftpub4b ftpub4b -- gpu ftpub4b ftpub4b "$F4"
row zs4b z4 -- gpu zs4b z4 $Z4
row ens_ftpub4b_ftpub_w05 ftpub4b z2 -- ens ens_ftpub4b_ftpub_w05 ftpub4b "$F4" z2 "$FP" 0.5
row ens_ftpub4b_ftpub_w025 ftpub4b z2 -- ens ens_ftpub4b_ftpub_w025 ftpub4b "$F4" z2 "$FP" 0.25
row ens_ftpub4b_ftpub_w1 ftpub4b z2 -- ens ens_ftpub4b_ftpub_w1 ftpub4b "$F4" z2 "$FP" 1
row ens_zs4b_zs06_w05 z4 z2 -- ens ens_zs4b_zs06_w05 z4 $Z4 z2 $Z 0.5
row z3 z3 -- gpu z3 z3 $Z
row bgem3_nolb bgem3 -- raw bgem3_nolb bgem3 BAAI/bge-reranker-v2-m3
row ftpub2x ftpub2x -- gpu ftpub2x ftpub2x "$Z+lora=models/ftpub2x"
row ftpub4b2x ftpub4b2x -- gpu ftpub4b2x ftpub4b2x "$Z4+lora=models/ftpub4b2x"
row ens_ftpub4b2x_ftpub_w05 ftpub4b2x z2 -- ens ens_ftpub4b2x_ftpub_w05 ftpub4b2x "$Z4+lora=models/ftpub4b2x" z2 "$FP" 0.5
row ens_ftpub4b2x_ftpub2x_w05 ftpub4b2x ftpub2x -- ens ens_ftpub4b2x_ftpub2x_w05 ftpub4b2x "$Z4+lora=models/ftpub4b2x" ftpub2x "$Z+lora=models/ftpub2x" 0.5
for s in ce_l6 ce_l12 ce_l12_512 ce_bgeb ce_bgeb512 ds_bgeb; do row $s $s -- gpu $s $s models/$s; done
for w in 025:0.25 05:0.5 1:1; do row ens_zs06_bgem3_w${w%%:*} z2 bgem3 -- ens ens_zs06_bgem3_w${w%%:*} z2 $Z bgem3 BAAI/bge-reranker-v2-m3 ${w##*:}; done
row c4b c4b -- gpu c4b c4b $Z4
ok=1; for r in $R5; do [ -s data/$r/list_lw4.json ] || ok=0; done
[ $ok = 1 ] && ! done_ lw4 && python $IR/list_builder/write_submissions.py lw lw4 z2 $Z lw4
for s in ftsyn_on ftsyn_lit ftsyn_lib fttrans ftreal_lib ftwho_lib; do row $s $s -- gpu $s $s "$Z+lora=models/$s"; done
for s in ft4b_on ft4b_lit ft4b_lib; do row $s $s -- gpu $s $s "$Z4+lora=models/$s"; done
python $IR/baselines/bm25_ctx_alone.py $R5
python $IR/aggregate/score_test_subs.py
