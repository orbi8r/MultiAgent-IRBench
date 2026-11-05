#!/bin/bash
# Lexical and trivial baselines on all eight runs, no training. Tags b3_*. CPU.
# EST=1500
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
python $IR/baselines/b3_lex.py
