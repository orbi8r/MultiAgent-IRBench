#!/bin/bash
# EST=1200
# NEEDS=40_cache4b2x
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
# CPU only: methods C and D over the 4B score cache, K200 and expansion lists
python $IR/structure_aware/sectional_rerank.py 4b 2>&1 | tail -30
