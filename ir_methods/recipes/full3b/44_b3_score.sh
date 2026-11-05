#!/bin/bash
# Score the b3 submissions with the repo scorer. CPU.
# EST=1200
# NEEDS=42_b3_lex 43_b3_dense
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
TASKGLOB='*_b3_*' python $IR/aggregate/score_submissions.py --subs b3subs --out-name b3scores
