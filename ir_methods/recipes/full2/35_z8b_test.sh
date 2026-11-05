#!/bin/bash
# The same 8B readers on the test runs, after 34_z8b_dev. GPU.
# EST=11000
# NEEDS=34_z8b_dev
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/reader_8b.py test
rm -rf $RC/z8b
