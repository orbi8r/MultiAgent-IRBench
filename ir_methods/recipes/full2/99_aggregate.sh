#!/bin/bash
# Join scores, rows and timing into results_full2.csv, results_full2.md and the summary. CPU.
# EST=900
# LAST=1
# NEEDS=98_score_all
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/aggregate/aggregate_full2.py
