"""Recall of every key free family: the share of atoms (the units each answer key credits, all query types) with any
credited piece in the family's top K. Keys are used only to measure. Reads data/<run>/fam_*.json from the working
directory and writes the recall table as JSON.
    python family_recall.py OUT.json   (RUNS env as everywhere)"""
import json, sys, glob, statistics as st
from collections import defaultdict
import os
os.environ.setdefault("IRBENCH_KEYS", "1")  # diagnostic: scores against the answer key
from run_loader import load, S, RUNS

KS = (10, 25, 50, 100, 200, 400, 1000, 2000)
REC = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # fam -> K -> run -> [bool]
for name in RUNS:
    d, run, key, qs, blocks, events = load(name)
    fams = {}
    for f in sorted(glob.glob(f"data/{name}/fam_*.json")):
        fams.update(json.load(open(f)))
    if "pass1_rrf" in fams:
        fams["fuse0"] = fams.pop("pass1_rrf")
    for q in qs:
        atoms = S.credits(key[q["qid"]], run)
        for fam, L in fams.items():
            p = {x: i for i, x in enumerate(L.get(q["qid"], []))}
            for a in atoms:
                r = min((p[x] for x in a if x in p), default=10 ** 6)
                for k in KS:
                    REC[fam][k][name].append(r < k)
out = {}
for fam in sorted(REC):
    out[fam] = {}
    for k in KS:
        allv = [v for n in REC[fam][k] for v in REC[fam][k][n]]
        out[fam][k] = dict(pooled=st.mean(allv), n=len(allv), runs={n: st.mean(v) for n, v in REC[fam][k].items()})
    print(f"{fam:16s}", " ".join(f"@{k}={out[fam][k]['pooled']:.3f}" for k in KS), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
