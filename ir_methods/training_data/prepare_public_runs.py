"""Pieces and training rows for a public pseudo run (no benchmark data, no keys).
    python prepare_public_runs.py pieces RUNDIR DATADIR     -> DATADIR/pieces.jsonl, questions.jsonl
    python prepare_public_runs.py rows DATADIR [CONVNEG]    -> DATADIR/train_synth.jsonl from DATADIR/synth.jsonl"""
import hashlib, json, sys
from contextual_pieces import build, toks
from bm25_tools import BM, rank

mode = sys.argv[1]
if mode == "pieces":
    rd, dd = sys.argv[2], sys.argv[3]
    import os
    os.makedirs(dd, exist_ok=True)
    ev = [json.loads(l) for l in open(f"{rd}/events.jsonl")]
    bl = [json.loads(l) for l in open(f"{rd}/blocks.jsonl")]
    with open(f"{dd}/pieces.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in build(ev, bl))
    open(f"{dd}/questions.jsonl", "w").close()
    sys.exit()
dd = sys.argv[2]
P = [json.loads(l) for l in open(f"{dd}/pieces.jsonl")]
ids = [p["id"] for p in P]
ctx = {p["id"]: p["ctx"] for p in P}
norm = lambda t: " ".join(t.split()).lower()
ntext = {p["id"]: norm(p["text"]) for p in P}
bm = BM([toks(p["ctx"]) for p in P])
agents = sorted({p["agent"] for p in P})
rows = []
for l in open(f"{dd}/synth.jsonl"):
    s = json.loads(l)
    tt, ans = ntext[s["target"]], norm(s["answer"])
    asker = agents[int(hashlib.md5(s["target"].encode()).hexdigest(), 16) % len(agents)]
    ok = lambda y: y != s["target"] and tt not in ntext[y] and ntext[y] not in tt and ans not in ntext[y]
    negs = [ctx[ids[i]] for i in rank(bm.scores(toks(s["question"])), 60) if ok(ids[i])][:40]
    if len(negs) >= 7:
        rows.append(dict(query=f"{s['question']} (asked by {asker})", pos=ctx[s["target"]], negs=negs))
with open(f"{dd}/train_synth.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(dd, len(rows), "rows")
