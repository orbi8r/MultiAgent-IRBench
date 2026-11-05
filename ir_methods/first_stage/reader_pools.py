"""Key free candidate pools for the test runs, in two rounds.
Round 1 (RND=1): the union of the top 200 of base (RRF of bm_ctx and bm_plain) and fuse0 (RRF of the seven pass 1 lists),
also saved as f1.json.
Round 2 (RND=2): reads the reader scores in data/<run>/<ADAPTERTAG or pub>sub1.json. For base and fuse0, and for each K
in 5, 10, 20, 30, 50, 100 and 200, the reader's top 5 are seeds, and the top 25 of each expansion list not yet scored
are added.
    python reader_pools.py 1|2 [ADAPTERTAG]  -> data/<run>/pool<RND>.json"""
import json, sys
from run_loader import RUNS
from bm25_tools import rrf
from arch1 import Run
from arch2 import expand

RND = int(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else "pub"
KS = (5, 10, 20, 30, 50, 100, 200)
for name in RUNS:
    r = Run(name)
    S1 = json.load(open(f"data/{name}/{TAG}sub1.json")) if RND == 2 else None
    out, f1 = {}, {"base": {}, "fuse0": {}}
    for q in r.qs:
        L, sc = r.pass1(q)
        firsts = {"base": rrf(L["bm_ctx"], L["bm_plain"]), "fuse0": rrf(*L.values())}
        if RND == 1:
            out[q["qid"]] = list(dict.fromkeys(r.ids[i] for l in firsts.values() for i in l[:200]))
            for k, l in firsts.items():
                f1[k][q["qid"]] = [r.ids[i] for i in l[:200]]
            continue
        s1 = S1[q["qid"]]; new = set()
        for lst in firsts.values():
            for K in KS:
                seeds = sorted(lst[:K], key=lambda i: -s1.get(r.ids[i], -1e9))[:5]
                for l in expand(r, q, seeds, sc).values():
                    new |= {r.ids[i] for i in l[:25] if r.ids[i] not in s1}
        out[q["qid"]] = sorted(new)
    json.dump(out, open(f"data/{name}/pool{RND}.json", "w"))
    if RND == 1:
        json.dump(f1, open(f"data/{name}/f1.json", "w"))
    print(name, RND, sum(len(v) for v in out.values()) / len(out), "per q", flush=True)
