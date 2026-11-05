#!/bin/bash
# EST=6000
# NEEDS=12_prefix_rows 10_prep_structure
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
# train the 0.6B on P2 prefixed rows (seed 1, ftpub recipe) and score the 3 dev runs only: the test sub is decided and written by 30c_sp06_pick
bash $IR/structure_aware/train_score_06b.sh P2 1 dev
