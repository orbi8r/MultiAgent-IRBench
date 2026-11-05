#!/bin/bash
# Reader configs of the fast pipeline written as submissions. Tags f2_fast_*. GPU.
# EST=4500
# NEEDS=10_prep
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/fast_pipeline/timed_run_configs.py K50,X50,X200,S50_20,S200_30
