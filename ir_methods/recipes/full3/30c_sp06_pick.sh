#!/bin/bash
# EST=1800
# NEEDS=30a_sp06_P1 30b_sp06_P2
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
# dev winner between P1 and P2, gate.json, freeze, then the 5 test subs of the winner only when the gate is go
python $IR/structure_aware/pick_prefix_style.py "P1 P2" 1
