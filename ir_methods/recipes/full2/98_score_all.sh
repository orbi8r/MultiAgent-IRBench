#!/bin/bash
# Score every frozen submission with the repo scorer. CPU.
# EST=1500
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/aggregate/score_submissions.py
