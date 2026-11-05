"""Reruns the full2 best pipeline (4B reader plus a saved LoRA over fuse0 top 200, then the top 25 of each expansion list) to keep
the per pair score cache for the 4B sectional work, and to record timing. No sub is written: the f2_ controls already exist.
    python cache_pool_scores_4b.py NAME      NAME = ftpub4b2x, adapter models/<NAME>; cache $RC/<NAME>/<run>.json = {qid: {piece_id: score rounded to 5 places}}
Resumes per run (marker $RC/<NAME>/<run>.ok); the cache itself also saves every 3000 pairs. Timing tag t3_<NAME>, rows and frozen file, no f2 tags."""
import os, sys
from pathlib import Path
import study_helpers, yes_no_scorer
from bm25_tools import rrf
from arch1 import Run
from arch2 import expand

name = sys.argv[1]
M4 = os.environ.get("M4", "Qwen/Qwen3-Reranker-4B")
RC = Path(os.environ.get("RCROOT", study_helpers.BASE / "rc"))
T = f"t3_{name}"
st = study_helpers.freeze(T, dict(reader=f"{M4} + LoRA models/{name}", first="fuse0 top 200", expansion_top=25, seeds=5, cache=f"rc/{name}/<run>.json",
                       note="timing only, scoring seconds without model load, the score cache is kept"))
study_helpers.row(T, "9.6 full3 timing and 4B score cache", f"{M4} adapter {name}, fuse0 top 200 plus the top 25 of each expansion list, scoring only", "PROBE",
       "fuse0 top 200 then expansion m25 as f2_ftpub4b2x_exp_fuse0_k200_m25, no sub", "timing only, no sub", "the per pair cache stays in rc for the full3 sectional rows")


class RCache(yes_no_scorer.Cache):
    def save(self):  # rounded floats keep the file small (about 12k pairs per run)
        study_helpers.atomic(self.path, {q: {x: round(v, 5) for x, v in d.items()} for q, d in self.d.items()})


sc = None
for split, run in study_helpers.plan():
    ok = RC / name / f"{run}.ok"
    if ok.exists():
        continue
    r = Run(run); qs = r.qs[:study_helpers.NQ]
    ctx = {p["id"]: p["ctx"] for p in r.P}; qtext = {q["qid"]: f"{q['question']} (asked by {q['asker']})" for q in qs}
    cache = RCache(RC / name / f"{run}.json")
    resumed = sum(len(v) for v in cache.d.values())
    if sc is None:
        sc = yes_no_scorer.Scorer(M4, adapter=str(study_helpers.BASE / "models" / name))
    s0 = sc.secs
    L1 = {q["qid"]: r.pass1(q) for q in qs}
    first = {q["qid"]: rrf(*L1[q["qid"]][0].values())[:200] for q in qs}
    cache.fill(sc, {k: [r.ids[i] for i in v] for k, v in first.items()}, qtext, ctx)
    add = {}
    for q in qs:
        qid = q["qid"]; c = first[qid]; s = cache.d[qid]
        Lx = expand(r, q, sorted(c, key=lambda i: -s[r.ids[i]])[:5], L1[qid][1])
        cs = set(c)
        add[qid] = list(dict.fromkeys(i for l in Lx.values() for i in l[:25] if i not in cs))
    cache.fill(sc, {k: [r.ids[i] for i in v] for k, v in add.items()}, qtext, ctx)
    cache.save()
    pairs = sum(len(v) for v in cache.d.values())
    study_helpers.timing(T, run, sc.secs - s0, sc.peak(), load=sc.load_s, pairs=pairs, resumed_pairs=resumed)
    ok.touch()
    study_helpers.log("done", split, run, "pairs", pairs, "cache bytes", os.path.getsize(cache.path))
