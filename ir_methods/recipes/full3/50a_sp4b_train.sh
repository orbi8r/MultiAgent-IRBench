#!/bin/bash
# EST=14000
# NEEDS=30c_sp06_pick 12_prefix_rows
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
# gated on 30c: the 4B reader trained once on the winning style prefixed rows, unseeded like ftpub4b, step checkpoints so a killed attempt resumes
read GO STYLE < <(python -c "import json,os;g=json.load(open(os.environ.get('GATEF') or os.environ['IRBENCH_WORK']+'/out/full3_30c_sp06_pick/gate.json'));print(int(g['go']),g['style'])")
[ -n "$GO" ] || { echo "gate.json unreadable"; exit 1; }
if [ "$GO" != 1 ]; then echo "gate no go, see out/full3_30c_sp06_pick/gate.json" > $IRBENCH_OUT/SKIPPED.txt; exit 0; fi
echo $STYLE > $IRBENCH_OUT/style.txt
STEPCKPT=${STEPCKPT:-400} python $IR/structure_aware/train_prefix_reader.py f3_sp4b_${STYLE}${TAGSUF} $STYLE none Qwen/Qwen3-Reranker-4B
ls $IRBENCH_WORK/adapters3/f3_sp4b_${STYLE}${TAGSUF} | head -3
