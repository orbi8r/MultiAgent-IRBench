#!/bin/bash
# Join the b3 rows, timing and scores into results_full3b.csv in $IRBENCH_OUT. CPU.
# EST=60
# NEEDS=44_b3_score
# LAST=1
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
python $IR/aggregate/aggregate_baselines.py
