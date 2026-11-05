"""Training rows for the reranker: {query, pos, negs}. Hard negatives are the contextual BM25 top 40 minus any piece that
could also answer (same or containing text, or holding the answer string).
    python make_train_rows.py synth RUN     -> data/RUN/train_synth.jsonl  (from data/RUN/synth.jsonl, no keys)
    python make_train_rows.py real RUN      -> data/RUN/train_real.jsonl   (from the run's answer key: use only for OTHER runs' folds)
"""
import json, sys, hashlib
import os
if sys.argv[1:2] == ["real"]:
    os.environ["IRBENCH_KEYS"] = "1"  # real mode reads the answer key of the folds it is given
from run_loader import load, S
from contextual_pieces import toks
from bm25_tools import BM, rank

mode, name = sys.argv[1], sys.argv[2]
d, run, key, qs, blocks, events = load(name)
P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
ids = [p["id"] for p in P]
ctx = {p["id"]: p["ctx"] for p in P}
norm = lambda t: " ".join(t.split()).lower()
ntext = {x: norm(run["text"][x]) for x in ids}
bm = BM([toks(p["ctx"]) for p in P])
agents = sorted({p["agent"] for p in P})
import os
CONVNEG = int(os.environ.get("CONVNEG", 0))  # hard negatives from the target's own conversation (brief, plan, code vs result)
conv_of = {e["id"]: e["conv"] for e in events}
conv_seq = {}
for k, e in enumerate(events):
    conv_seq.setdefault(e["conv"], []).append(e["id"])
by_ev = {}
for p in P:
    by_ev.setdefault(p["event"], []).append(p["id"])


def conv_negs(target, n, window=8):
    e = target.split("/")[0]
    seq = conv_seq[conv_of[e]]
    i = seq.index(e)
    near = sorted(seq[max(0, i - window):i + window + 1], key=lambda x: abs(seq.index(x) - i))
    return [y for x in near for y in by_ev.get(x, [])][: n * 3]
rows = []
if mode.startswith("synth"):
    for l in open(f"data/{name}/{mode}.jsonl"):
        s = json.loads(l)
        tt, ans = ntext[s["target"]], norm(s["answer"])
        asker = agents[int(hashlib.md5(s["target"].encode()).hexdigest(), 16) % len(agents)]
        q = f"{s['question']} (asked by {asker})"
        ok = lambda y: y != s["target"] and tt not in ntext[y] and ntext[y] not in tt and ans not in ntext[y]
        cn = [y for y in conv_negs(s["target"], CONVNEG) if ok(y)][:CONVNEG] if CONVNEG else []
        negs = [ctx[y] for y in cn] + [ctx[ids[i]] for i in rank(bm.scores(toks(s["question"])), 60) if ok(ids[i]) and ids[i] not in cn][:40 - len(cn)]
        if len(negs) >= 7:
            rows.append(dict(query=q, pos=ctx[s["target"]], negs=negs))
else:
    for q in qs:
        atoms = S.credits(key[q["qid"]], run)
        credited = set().union(*atoms)
        negs = [ctx[ids[i]] for i in rank(bm.scores(toks(q["question"])), 80) if ids[i] not in credited][:40]
        for a, raw in zip(atoms, key[q["qid"]]["atoms"]):
            firsts = [x for x, m in raw.items() if m >= 1.0 and x in ctx]
            for x in sorted(firsts, key=lambda x: run["tokens"][x])[:2]:
                rows.append(dict(query=f"{q['question']} (asked by {q['asker']})", pos=ctx[x], negs=negs))
out = f"data/{name}/train_{mode}{os.environ.get('OTAG', '')}.jsonl"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(out, len(rows))
