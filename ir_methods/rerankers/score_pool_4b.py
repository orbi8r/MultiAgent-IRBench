"""Scores the fuse0 top 200 and its expansion lists with a saved 4B reader adapter.
Writes the tags f2_<NAME>_fuse0_k200 and f2_<NAME>_exp_fuse0_k200_m25.
    python score_pool_4b.py NAME      NAME = ftpub4b or ftpub4b2x, adapter models/<NAME>, base Qwen3 Reranker 4B (env M4)
The expansion seeds are the top 5 of the fuse0 top 200 under the 4B reader, expanded the same way as write_pool_submissions.py's pub step. Scores are cached in $RC/<NAME>/<run>.json. No key is read."""
import os, sys
from pathlib import Path
import study_helpers, tags, yes_no_scorer
from bm25_tools import rrf
from arch1 import Run
from arch2 import expand

name = sys.argv[1]
M4 = os.environ.get("M4", "Qwen/Qwen3-Reranker-4B")
RC = Path(os.environ.get("RCROOT", study_helpers.BASE / "rc"))
T1, T2 = f"f2_{name}_fuse0_k200", f"f2_{name}_exp_fuse0_k200_m25"
for t, m in ((T1, 0), (T2, 25)):
    st = study_helpers.freeze(t, dict(reader=f"{M4} + LoRA models/{name}", first="fuse0 top 200", expansion_top=m, seeds=5, builder="build_list.build SCAN 30 EARLIEST"))
    tags.row(t, "FIXED", f"{M4} adapter {name}, fuse0 top 200, expansion top {m} of each list; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0",
             "dev decides nothing, declared")
sc = None
for split, run in study_helpers.plan():
    if study_helpers.have(T1, split, run) and study_helpers.have(T2, split, run):
        continue
    r = Run(run); F = study_helpers.from_run(r); qs = r.qs[:study_helpers.NQ]
    ctx = {p["id"]: p["ctx"] for p in r.P}; qtext = {q["qid"]: f"{q['question']} (asked by {q['asker']})" for q in qs}
    cache = yes_no_scorer.Cache(RC / name / f"{run}.json")
    if sc is None:
        sc = yes_no_scorer.Scorer(M4, adapter=str(study_helpers.BASE / "models" / name))
    L1 = {q["qid"]: r.pass1(q) for q in qs}
    first = {q["qid"]: rrf(*L1[q["qid"]][0].values())[:200] for q in qs}
    cache.fill(sc, {k: [r.ids[i] for i in v] for k, v in first.items()}, qtext, ctx)
    sub1, sub2, add, t0 = {}, {}, {}, sc.secs
    for q in qs:
        qid = q["qid"]; c = first[qid]; s = cache.d[qid]
        sub1[qid] = study_helpers.build(F, q, r.run, [r.ids[i] for i in c], [s[r.ids[i]] for i in c])
        Lx = expand(r, q, sorted(c, key=lambda i: -s[r.ids[i]])[:5], L1[qid][1])
        cs = set(c)
        add[qid] = list(dict.fromkeys(i for l in Lx.values() for i in l[:25] if i not in cs))
    study_helpers.put(T1, split, run, sub1, qs)
    cache.fill(sc, {k: [r.ids[i] for i in v] for k, v in add.items()}, qtext, ctx)
    for q in qs:
        qid = q["qid"]; c = first[qid] + add[qid]
        sub2[qid] = study_helpers.build(F, q, r.run, [r.ids[i] for i in c], [cache.d[qid][r.ids[i]] for i in c])
    study_helpers.put(T2, split, run, sub2, qs)
    study_helpers.log("done", split, run, "pairs", sc.pairs)
