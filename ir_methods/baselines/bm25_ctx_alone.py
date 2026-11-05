"""Tag bm25_ctx_alone: the contextual BM25 list on its own, no reader and no list builder.
Inputs: data/<run>/lex.json from the first stage candidate script (the ctx list of every question, top 300).
Output: $IRBENCH_OUT/subs/bm25_ctx_alone/<run>.json, a submission {qid: [piece ids]}.
    python bm25_ctx_alone.py [run ...]     (default: the runs in $RUNS)"""
import json, os, sys
from pathlib import Path
from run_loader import RUNS

OUT = Path(os.environ["IRBENCH_OUT"]) / "subs" / "bm25_ctx_alone"
OUT.mkdir(parents=True, exist_ok=True)
for run in (sys.argv[1:] or RUNS):
    lex = json.load(open(f"data/{run}/lex.json"))
    json.dump({q: v["ctx"] for q, v in lex.items()}, open(OUT / f"{run}.json", "w"))
    print(run, len(lex), "questions")
