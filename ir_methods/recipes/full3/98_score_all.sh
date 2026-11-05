#!/bin/bash
# Score every frozen submission of the structure aware study with the repo scorer. CPU.
# EST=1500
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
python $IR/aggregate/score_submissions.py
