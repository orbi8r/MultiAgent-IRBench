"""Write the algorithmic first stage (fuse0: RRF of the arch1 pass 1 lists) top 100 per question and time it per run.
    python write_fuse0.py -> data/<run>/fuse0.json, fuse0_time.json"""
import json, time
from run_loader import RUNS
from bm25_tools import rrf
from arch1 import Run
T = {}
for name in RUNS:
    t0 = time.time(); r = Run(name); t1 = time.time()
    out = {q["qid"]: [r.ids[i] for i in rrf(*r.pass1(q)[0].values())[:100]] for q in r.qs}
    t2 = time.time()
    T[name] = dict(index_s=round(t1 - t0, 2), queries_s=round(t2 - t1, 2), n_q=len(r.qs), stages={k: round(v, 2) for k, v in r.T.items()})
    json.dump(out, open(f"data/{name}/fuse0.json", "w"))
    json.dump(T[name], open(f"data/{name}/fuse0_time.json", "w"))
print(json.dumps(T))
