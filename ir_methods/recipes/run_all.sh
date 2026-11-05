#!/bin/bash
# Generic local runner: run every recipe of one folder in numeric order and skip the ones that already finished.
#   bash recipes/run_all.sh full3b            all recipes of recipes/full3b
#   bash recipes/run_all.sh full2 30 99       only recipes whose number is between 30 and 99
# A finished recipe leaves $IRBENCH_WORK/out/.done/<folder>_<recipe name>. Delete that file to run a recipe again.
# Lines such as "# NEEDS=10_prep" inside a recipe name the recipes that have to finish first, "# EST=3000" is the
# estimated run time in seconds on one 11 GB GPU. Run it with nohup or inside tmux, a full2 pass takes many hours.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="${1%/}"; lo="${2:-0}"; hi="${3:-999}"
source "$here/../env.sh"
for f in $(ls "$here/$dir"/[0-9]*.sh | sort); do
  n="$(basename "$f" .sh)"; num="${n%%_*}"; num="${num//[!0-9]/}"
  [ "$((10#$num))" -lt "$lo" ] || [ "$((10#$num))" -gt "$hi" ] && continue
  mark="$IRBENCH_WORK/out/.done/${dir}_$n"
  [ -f "$mark" ] && { echo "skip $dir/$n"; continue; }
  echo "run $dir/$n"
  bash "$f" || { echo "FAILED $dir/$n"; exit 1; }
  mkdir -p "$IRBENCH_WORK/out/.done"; touch "$mark"
done
