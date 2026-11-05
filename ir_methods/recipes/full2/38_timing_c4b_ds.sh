#!/bin/bash
# The c4b cascade, the ds_bgeb student over z2 and the timing only tags t2_c4b, t2_ds_bgeb, t2_ftpub_k200. GPU.
# EST=3300
# NEEDS=30_z2_base
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
set -e
python $IR/rerankers/score_z2.py z2ds "f2_ds_bgeb|models/ds_bgeb|build"
python $IR/fast_pipeline/cascade_timing.py c4b
python $IR/fast_pipeline/cascade_timing.py t2
python $IR/fast_pipeline/cascade_timing.py fast
rm -rf $WORK/tim
