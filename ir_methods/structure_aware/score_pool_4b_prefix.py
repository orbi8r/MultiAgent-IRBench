"""Pooled 4B scoring as in cache_pool_scores_4b.py, with a trained prefix reader. The 4B adapter trained on prefixed rows (train_prefix_reader.py) reads the prefixed candidate text
(data/<run>/f3_pfx_<style>.json) over the fuse0 top 200 (tag f3_sp4b_<style>) and with the expansion m25 (tag f3_sp4b_<style>_exp).
    python score_pool_4b_prefix.py STYLE     adapter $IRBENCH_WORK/adapters3/f3_sp4b_<STYLE>, cache $RC/f3_sp4b_<STYLE>/<run>.json (removed by the task)
Order: dev runs, then both tags are frozen (declared settings plus the dev numbers against the f2 controls as evidence), then the test runs once."""
import json, os, sys
from pathlib import Path
import study_helpers, yes_no_scorer, dev_gate
from bm25_tools import rrf
from arch1 import Run
from arch2 import expand

style = sys.argv[1]
name = f"f3_sp4b_{style}{os.environ.get('TAGSUF', '')}"  # the suffix is for smokes only
M4 = os.environ.get("M4", "Qwen/Qwen3-Reranker-4B")
RC = Path(os.environ.get("RCROOT", study_helpers.BASE / "rc"))
T1, T2 = name, f"{name}_exp"
CT = {T1: "f2_ftpub4b_fuse0_k200", T2: "f2_ftpub4b_exp_fuse0_k200_m25"}
SET = {t: dict(reader=f"{M4} + LoRA adapters3/{name}", first="fuse0 top 200", expansion_top=m, seeds=5, prefix_style=style,
               train=f"4B on {style} prefixed rows of pub_00 to pub_14, 4964 rows, 1 epoch, unseeded as ftpub4b, train_reranker.py", builder="build_list.build SCAN 30 EARLIEST")
       for t, m in ((T1, 0), (T2, 25))}
for t in (T1, T2):
    study_helpers.row(t, "9.3 full3 trained prefix 4B", f"{M4} trained on {style} prefixed public rows, fuse0 top 200" + (" plus the top 25 of each expansion list" if t == T2 else "")
           + "; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0", "FIXED", json.dumps(SET[t]), "dev decides nothing, the style comes from the 0.6B gate", f"control {CT[t]}")
sc, frozen = None, False
for split, run in study_helpers.plan():
    if split == "test" and not frozen:
        ev = {}
        for t in (T1, T2):
            try:
                ev[t] = dev_gate.gate(t, CT[t])
            except Exception as e:  # the dev numbers are evidence only here, a missing control must not stop the test run
                ev[t] = dict(why=f"dev comparison unavailable: {e!r}")
        study_helpers.atomic(study_helpers.OUT / "dev_vs_control.json", ev, indent=1)
        for t in (T1, T2):
            study_helpers.freeze(t, SET[t], dict(dev=ev[t]["why"]))
        frozen = True
    if study_helpers.have(T1, split, run) and study_helpers.have(T2, split, run):
        continue
    r = Run(run); F = study_helpers.from_run(r); qs = r.qs[:study_helpers.NQ]
    ctx = json.load(open(f"data/{run}/f3_pfx_{style}.json", encoding="utf-8"))
    qtext = {q["qid"]: f"{q['question']} (asked by {q['asker']})" for q in qs}
    cache = yes_no_scorer.Cache(RC / name / f"{run}.json")
    if sc is None:
        sc = yes_no_scorer.Scorer(M4, adapter=str(study_helpers.BASE / "adapters3" / name))
    s0 = sc.secs
    L1 = {q["qid"]: r.pass1(q) for q in qs}
    first = {q["qid"]: rrf(*L1[q["qid"]][0].values())[:200] for q in qs}
    cache.fill(sc, {k: [r.ids[i] for i in v] for k, v in first.items()}, qtext, ctx)
    s1 = sc.secs - s0
    sub1, sub2, add = {}, {}, {}
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
    study_helpers.timing(T1, run, s1, sc.peak(), load=sc.load_s, pairs=len(qs) * 200)
    study_helpers.timing(T2, run, sc.secs - s0, sc.peak(), load=sc.load_s, pairs=sum(len(v) for v in cache.d.values()))
    study_helpers.log("done", split, run, "pairs", sc.pairs)
