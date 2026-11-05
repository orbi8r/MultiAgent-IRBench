"""Key free part of arch2 only: round 0 lists fused, two feedback rounds, RRF. The LambdaMART of arch1 and arch2
is not run, because it is trained on the answer keys of the other runs, which would be the test runs here.
    python arch2_rrf.py -> data/<run>/fam_arch2.json {arch2_rrf, pass1_rrf}, arch2_rrf_time.json"""
import json, time
from run_loader import RUNS
from bm25_tools import rrf
from arch1 import Run, N
from arch2 import expand, ROUNDS, SEEDS
T = {}
for n in RUNS:
    t0 = time.time(); r = Run(n); t1 = time.time()
    out = {"arch2_rrf": {}, "pass1_rrf": {}}
    for q in r.qs:
        L, sc = r.pass1(q)
        cur = rrf(*L.values()); out["pass1_rrf"][q["qid"]] = [r.ids[i] for i in cur[:N]]
        allL = dict(L)
        for rd in range(ROUNDS):
            E = expand(r, q, cur[:SEEDS], sc)
            allL.update({f"{k}{rd}": v for k, v in E.items()})
            cur = rrf(*allL.values())
        out["arch2_rrf"][q["qid"]] = [r.ids[i] for i in cur[:N]]
    t2 = time.time()
    T[n] = dict(index_s=round(t1 - t0, 2), queries_s=round(t2 - t1, 2), n_q=len(r.qs), stages={k: round(v, 2) for k, v in r.T.items()})
    json.dump(out, open(f"data/{n}/fam_arch2.json", "w"))
json.dump(T, open("arch2_rrf_time.json", "w"), indent=1); print(json.dumps(T))
