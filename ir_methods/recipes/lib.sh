# Sourced first by every recipe. Sets the run lists, the task output folder, the model names and the cwd.
# Optional overrides: RUNSEL (runs to process), DEVR and TESTR (runs scored per split, an empty value means none), M4 M8 LW4 LW8.
_r="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_f="$(cd "$(dirname "${BASH_SOURCE[1]:-.}")" && pwd)"; _s="$(basename "$_f")"; [ "$_s" = recipes ] && _s=""
source "$_r/../env.sh"
# The task name is the study folder plus the recipe name, so two studies with a 99_aggregate.sh keep separate output folders.
export TASKNAME="${TASKNAME:-${_s:+${_s}_}$(basename "${BASH_SOURCE[1]:-task}" .sh)}"
unset _f _s
export IRBENCH_OUT="$IRBENCH_WORK/out/$TASKNAME"
export DEV="online_shop library_maintainers literature_review"
export TEST="on_call due_diligence executive_assistants help_desk textbook_team"
export ALL="${RUNSEL:-$DEV $TEST}"
export DEVR="${DEVR-$DEV}" TESTR="${TESTR-$TEST}"
export SCR="$IRBENCH_SCRATCH" WORK="$IRBENCH_SCRATCH/work" RC="$IRBENCH_WORK/rc"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# test5/lib.sh sources this file before it sets its own TASKNAME, so it makes its own output folder.
[ "$(basename "${BASH_SOURCE[1]}")" = lib.sh ] || mkdir -p "$IRBENCH_OUT"
mkdir -p "$WORK" "$RC"
cd "$IRBENCH_WORK" || exit 1
export M4="${M4:-Qwen/Qwen3-Reranker-4B}" M8="${M8:-Qwen/Qwen3-Reranker-8B}" LW4="${LW4:-Qwen/Qwen3-4B-Instruct-2507}" LW8="${LW8:-Qwen/Qwen3-8B}"
export B06=Qwen/Qwen3-Reranker-0.6B
