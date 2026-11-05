#!/bin/bash
# EST=6000
# NEEDS=30c_sp06_pick
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
# gated on 30c: the winning style, seed 3 (control f2_ftpub_s3), dev subs then frozen settings then test subs
read GO STYLE < <(python -c "import json,os;g=json.load(open(os.environ.get('GATEF') or os.environ['IRBENCH_WORK']+'/out/full3_30c_sp06_pick/gate.json'));print(int(g['go']),g['style'])")
[ -n "$GO" ] || { echo "gate.json unreadable"; exit 1; }
if [ "$GO" != 1 ]; then echo "gate no go, see out/full3_30c_sp06_pick/gate.json" > $IRBENCH_OUT/SKIPPED.txt; exit 0; fi
bash $IR/structure_aware/train_score_06b.sh $STYLE 3 both freeze
