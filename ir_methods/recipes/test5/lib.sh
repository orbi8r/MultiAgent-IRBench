# Sourced by the test5 recipes: the frozen test study of the five test runs, one shared output folder out/test5.
# Helpers gs and gc run gpu_score.py once per run and tag and leave a sentinel data/<run>/.ok_<tag>.
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
export TASKNAME=test5 IRBENCH_OUT="$IRBENCH_WORK/out/test5"; mkdir -p "$IRBENCH_OUT"
R5="$TEST"
export RUNS="${R5// /,}"
export NO_TRITON=1 TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TOKENIZERS_PARALLELISM=false
# gs RUN TAG EMBEDDER RERANKER[,RERANKER] [K]
gs() { local run=$1 tag=$2; [ -e data/$run/.ok_$tag ] && { echo skip $run $tag; return 0; }
  python $IR/rerankers/gpu_score.py "$@" && touch data/$run/.ok_$tag; }
# gc RUN TAG RERANKER: reuse the frozen z2 candidate list instead of running the dense stage again
gc() { CANDS_FROM=data/$1/gpu_z2.json gs $1 $2 none "$3" 200; }
