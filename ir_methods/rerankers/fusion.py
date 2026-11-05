"""First stage fusion ablation, read by the ftpub reranker (a Qwen3-Reranker-0.6B LoRA adapter trained by train_ftpub.py), plus a longer expansion grid on fuse0.
Variants drop or keep some of the seven lists of the first pass (arch1.Run.pass1) before RRF. The top 100 of each variant is read by ftpub (tags end in _k100).
The top 5 by reader score seed the arch2.expand lists, and the top 25 of each are read too (tags end in _k100_m25), the same way write_pool_submissions.py's pub step does.
Also fuse0 top 200 with the top 50 and top 100 of each expansion list (f2_exp_fuse0_k200_m50 and m100).
Reader scores: data/<run>/ftpub_pool.json plus what is missing, cached in $RC/ftpub06/<run>.json. No key is read.
    python fusion.py"""
import os
from pathlib import Path
import study_helpers, tags, yes_no_scorer
from bm25_tools import rrf
from arch1 import Run
from arch2 import expand

NAMES = ["bm_ctx", "bm_plain", "bm_raw", "bm_nov", "rm3", "qdecomp", "qtype"]
RC = Path(os.environ.get("RCROOT", study_helpers.BASE / "rc"))
VT = {}  # tag -> variant settings
for v in tags.FL:
    keep = NAMES[:int(v[5:])] if v.startswith("first") else [n for n in NAMES if n != v[3:]]
    VT[f"f2_fl_{v}_k100"] = dict(lists=keep, K=100, M=0)
    VT[f"f2_fl_{v}_k100_m25"] = dict(lists=keep, K=100, M=25)
for m in (50, 100):
    VT[f"f2_exp_fuse0_k200_m{m}"] = dict(lists=NAMES, K=200, M=m)
FR = {}
for t, s in VT.items():
    FR[t] = study_helpers.freeze(t, dict(reader="ftpub 0.6B LoRA models/ftpub", rrf_k=60, seeds=5, builder="build_list.build SCAN 30 EARLIEST", **s))
    tags.row(t, "FIXED", f"first stage lists {','.join(FR[t]['lists'])}, top {FR[t]['K']} read by ftpub, expansion top {FR[t]['M']} of each list; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0",
             "dev decides nothing, the variants are declared and all are scored")
sc = None
runs_seen = {}
for split, run in study_helpers.plan():
    todo = [t for t in VT if not study_helpers.have(t, split, run)]
    if not todo:
        continue
    if run not in runs_seen:
        runs_seen.clear(); runs_seen[run] = Run(run)
    r = runs_seen[run]; F = study_helpers.from_run(r); qs = r.qs[:study_helpers.NQ]
    ctx = {p["id"]: p["ctx"] for p in r.P}; qtext = {q["qid"]: f"{q['question']} (asked by {q['asker']})" for q in qs}
    cache = yes_no_scorer.Cache(RC / "ftpub06" / f"{run}.json", seeds=[f"data/{run}/ftpub_pool.json"])
    L1 = {q["qid"]: r.pass1(q) for q in qs}
    firsts = {}
    for t in todo:
        s = FR[t]
        for q in qs:
            L, _ = L1[q["qid"]]
            firsts[(t, q["qid"])] = rrf(*[L[n] for n in s["lists"]])[:s["K"]]
    if sc is None:
        sc = yes_no_scorer.Scorer("Qwen/Qwen3-Reranker-0.6B", adapter=str(study_helpers.BASE / "models" / "ftpub"))
    cache.fill(sc, {q["qid"]: [r.ids[i] for (t, qid), c in firsts.items() if qid == q["qid"] for i in c] for q in qs}, qtext, ctx)
    exp = {}
    for (t, qid), c in firsts.items():
        s = FR[t]
        if not s["M"]:
            continue
        q = next(x for x in qs if x["qid"] == qid)
        order = sorted(c, key=lambda i: -cache.d[qid][r.ids[i]])
        Lx = expand(r, q, order[:5], L1[qid][1])
        cs = set(c)
        exp[(t, qid)] = list(dict.fromkeys(i for l in Lx.values() for i in l[:s["M"]] if i not in cs))
    cache.fill(sc, {q["qid"]: [r.ids[i] for (t, qid), a in exp.items() if qid == q["qid"] for i in a] for q in qs}, qtext, ctx)
    for t in todo:
        sub = {}
        for q in qs:
            qid = q["qid"]; c = firsts[(t, qid)] + exp.get((t, qid), [])
            sub[qid] = study_helpers.build(F, q, r.run, [r.ids[i] for i in c], [cache.d[qid][r.ids[i]] for i in c])
        study_helpers.put(t, split, run, sub, qs)
    study_helpers.log("done", split, run, len(todo), "tags")
