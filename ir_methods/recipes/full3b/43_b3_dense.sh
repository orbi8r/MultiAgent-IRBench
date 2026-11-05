#!/bin/bash
# Dense baselines on all eight runs, no training. Tags b3_*. GPU.
# EST=4500
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
python $IR/baselines/b3_dense.py
