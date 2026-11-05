#!/bin/bash
# Join scores, rows and timing of the structure aware study into results_full3.csv, results_full3.md and the summary. CPU.
# EST=900
# LAST=1
# NEEDS=98_score_all
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
python $IR/aggregate/aggregate_full3.py
