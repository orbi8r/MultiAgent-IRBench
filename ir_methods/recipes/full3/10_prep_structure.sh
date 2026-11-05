#!/bin/bash
# EST=900
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e -o pipefail
mkdir -p $IRBENCH_OUT
# prefixed context files per style and run (skips styles already written), then structure and token stats
for r in $ALL; do python $IR/structure_aware/write_prefixes.py $r; done
python $IR/structure_aware/structure_stats.py $IRBENCH_OUT/struct_stats.json $ALL
tail -c 600 $IRBENCH_OUT/struct_stats.json
