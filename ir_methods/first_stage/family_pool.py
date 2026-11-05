"""Candidate pool for the reader. Takes the top 200 of every family in fam_cpu.json and fam_arch2.json (arch2_rrf and
pass1_rrf among them) and the union of those per question.
    python family_pool.py -> data/<run>/fam5_lists.json {family: {qid: [ids]}}, data/<run>/fam5_pool.json {qid: [ids]}"""
import json
from run_loader import RUNS
for name in RUNS:
    fam = json.load(open(f"data/{name}/fam_cpu.json"))
    fam.update(json.load(open(f"data/{name}/fam_arch2.json")))
    lists = {f: {q: l[:200] for q, l in d.items()} for f, d in fam.items()}
    pool = {}
    for d in lists.values():
        for q, l in d.items():
            pool.setdefault(q, {}).update(dict.fromkeys(l))
    pool = {q: list(d) for q, d in pool.items()}
    json.dump(lists, open(f"data/{name}/fam5_lists.json", "w"))
    json.dump(pool, open(f"data/{name}/fam5_pool.json", "w"))
    print(name, len(lists), "families", sum(len(v) for v in pool.values()) / len(pool), "pieces per q", flush=True)
