#!/bin/bash
# EST=1800
# NEEDS=10_prep_structure
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
# prefixed public rows per style from the staged pub runs, after the flat rebuild is compared with pubrows (flatcheck.json in $IRBENCH_OUT)
[ -f $IRBENCH_OUT/rows_done ] || { python $IR/structure_aware/prefix_training_rows.py ${PUBSTAGE:-$IRBENCH_WORK/pubstage} $IRBENCH_WORK/pubrows ${ROWS3:-$IRBENCH_WORK/pubrows3} "${STYLES:-P1 P2}" 2>&1 | tail -40; touch $IRBENCH_OUT/rows_done; }
for s in ${STYLES:-P1 P2}; do ls ${ROWS3:-$IRBENCH_WORK/pubrows3}/$s | wc -l; done
# the staging is only needed here: drop it once the rows exist (a smoke narrows PUBSEL and keeps it)
[ -z "$PUBSEL" ] && rm -rf $IRBENCH_WORK/pubstage
du -sh ${ROWS3:-$IRBENCH_WORK/pubrows3} | tail -1
