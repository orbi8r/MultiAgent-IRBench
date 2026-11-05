"""Writes every grid cell as a submission. INTERNAL_SCORE=1 also prints the score table.
Pseudo relevance across the conversation for the BM25 reader (nothing fitted). A piece's score is its own z scored
BM25 plus B times the best z score among the other pieces of its event and the events up to w steps either side in the
same conversation (w = 0, 2 or 5). The support comes from the neighbourhood: the question's other words sit in the plan,
call or narration next to the answer. Grid shown in sample; B=0 is the BM25 reader.
    python conversation_prf.py"""
import json, statistics as st
from collections import defaultdict
import numpy as np
import os
SCORE = os.environ.get("INTERNAL_SCORE") == "1"
if SCORE:
    os.environ["IRBENCH_KEYS"] = "1"  # answer keys are opened only for the diagnostic score table
from run_loader import load, S, RUNS
from contextual_pieces import toks
from bm25_tools import BM
from file_provenance import Files
import build_list

build_list.SCAN, build_list.EARLIEST = 30, True
GRID = [(b, w) for b in (0, 0.2, 0.4, 0.8) for w in (0, 2, 5)]
R = defaultdict(list)
SUB = {}
SUBDIR = os.environ.get("SUBDIR", "subs_convprf")
for name in RUNS:
    d, run, key, qs, blocks, events = load(name)
    P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
    ids = [p["id"] for p in P]; pos = {x: i for i, x in enumerate(ids)}
    F = Files(events, P); F.by_event = defaultdict(list); F.tokens = run["tokens"]
    for p in P:
        F.by_event[p["event"]].append(p["id"])
    conv = defaultdict(list)
    for e in events:
        conv[e["conv"]].append(e["id"])
    cpos = {eid: (c, k) for c, l in conv.items() for k, eid in enumerate(l)}
    bc = BM([toks(p["ctx"]) for p in P])
    base = json.load(open(f"data/{name}/fam_cpu.json"))["base"]
    for q in qs:
        cands = base[q["qid"]][:200]; atoms = S.credits(key[q["qid"]], run) if SCORE else None
        sc = bc.scores(toks(q["question"]))
        allv = np.array(list(sc.values())) if sc else np.zeros(1)
        mu, sd = allv.mean(), allv.std() + 1e-9
        z = lambda i: (sc.get(i, 0) - mu) / sd
        for b, w in GRID:
            s = []
            for x in cands:
                c, k = cpos[x.split("/")[0]]
                nb = [pos[y] for kk in range(max(0, k - w), min(len(conv[c]), k + w + 1)) for y in F.by_event.get(conv[c][kk], []) if y != x]
                s.append(z(pos[x]) + b * max((z(i) for i in nb), default=0))
            lst = build_list.build(F, q, cands, s, run["text"])
            SUB.setdefault((name, b, w), {})[q["qid"]] = lst
            if SCORE:
                R[(b, w)].append(S.score_query(key[q["qid"]], lst, run, credit=atoms)["score"])
                R[(b, w, name)].append(R[(b, w)][-1])
    print(name, flush=True)
for (name, b, w), sub in SUB.items():
    d_ = f"{SUBDIR}/convprf_b{str(b).replace('.', 'd')}_w{w}"
    os.makedirs(d_, exist_ok=True)
    json.dump(sub, open(f"{d_}/{name}.json", "w"))
if SCORE:
    for g in GRID:
        print(g, "%.3f" % st.mean(R[g]), " ".join(f"{n[:6]}={st.mean(R[(*g, n)]):.3f}" for n in RUNS))
