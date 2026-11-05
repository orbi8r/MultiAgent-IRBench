"""Writes the contextual pieces and the lexical first stage lists for each run, as input to the GPU stage. No keys read.
Input: the runs as a comma separated argument, or every run in RUNS when omitted.
Output: data/<run>/pieces.jsonl, lex.json (top 300 per question from the ctx and plain indexes) and questions.jsonl."""
import json, os, sys
from run_loader import load, RUNS
from contextual_pieces import build, toks
from bm25_tools import BM, rank

OUT = "data"  # relative to the cwd, which is $IRBENCH_WORK
for name in (sys.argv[1].split(",") if len(sys.argv) > 1 else RUNS):
    d, run, key, qs, blocks, events = load(name)
    P = build(events, blocks)
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    with open(f"{OUT}/{name}/pieces.jsonl", "w") as f:
        for p in P:
            f.write(json.dumps(p) + "\n")
    ids = [p["id"] for p in P]
    plain = BM([toks(p["text"]) for p in P])
    ctx = BM([toks(p["ctx"]) for p in P])
    out = {}
    for q in qs:
        qt = toks(q["question"])
        out[q["qid"]] = dict(ctx=[ids[i] for i in rank(ctx.scores(qt), 300)], plain=[ids[i] for i in rank(plain.scores(qt), 300)])
    json.dump(out, open(f"{OUT}/{name}/lex.json", "w"))
    with open(f"{OUT}/{name}/questions.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    print(name, len(P), "pieces")
