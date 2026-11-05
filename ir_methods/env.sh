# Source this file from a shell before running anything in ir_methods:  source ir_methods/env.sh
# It sets the path variables and puts every stage folder on PYTHONPATH.
IR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IR
export IRBENCH_REPO="$(dirname "$IR")"
export IRBENCH_RUNS="${IRBENCH_RUNS:-$IRBENCH_REPO/runs}"
export IRBENCH_WORK="${IRBENCH_WORK:-$IR/work}"
export IRBENCH_OUT="${IRBENCH_OUT:-$IRBENCH_WORK/out/default}"
export IRBENCH_SCRATCH="${IRBENCH_SCRATCH:-$IRBENCH_WORK/scratch}"
mkdir -p "$IRBENCH_WORK/data" "$IRBENCH_WORK/models" "$IRBENCH_SCRATCH"
_pp="$IRBENCH_REPO"
for d in common first_stage baselines rerankers students fast_pipeline training_data list_builder expansion algorithmic rl_style structure_aware aggregate; do
  _pp="$_pp:$IR/$d"
done
export PYTHONPATH="$_pp${PYTHONPATH:+:$PYTHONPATH}"
unset _pp d
