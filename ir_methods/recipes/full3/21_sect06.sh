#!/bin/bash
# EST=900
# NEEDS=10_prep_structure
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
# CPU only: methods C and D over the cached ftpub 0.6B scores of z2 K200
python $IR/structure_aware/sectional_rerank.py 06 2>&1 | tail -30
