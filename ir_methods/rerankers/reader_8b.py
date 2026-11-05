"""Zero shot Qwen3 Reranker 8B over the top 200 of the z2 candidates (tag f2_z8b), and the cascade that rereads the top 50 of the 0.6B reranker's ranking (tag f2_c8b).
    python reader_8b.py dev|test      dev = the dev run (decides nothing, declares the memory mode), test = the test run (copies the dev run's frozen file)
Base scores come from data/<run>/gpu_z2s.json. 8B pair scores are cached in $RC/z8b/<run>.json, so a retry resumes.
Memory mode: nf4 first, fp16 with CPU offload when nf4 cannot load; the choice is written to the frozen file once."""
import json, os, shutil, sys
from pathlib import Path
import torch
import study_helpers, tags, yes_no_scorer

split = sys.argv[1]
BIG = os.environ.get("M8", "Qwen/Qwen3-Reranker-8B")
ZS06 = "Qwen/Qwen3-Reranker-0.6B"
RC, PREV = Path(os.environ.get("RCROOT", study_helpers.BASE / "rc")), Path(os.environ.get("PREVOUT", study_helpers.BASE / "out" / "34_z8b_dev"))
TAGS = ("f2_z8b", "f2_c8b")
runs = [r for s, r in study_helpers.plan() if s == split]

if split == "test":  # settings and timing come from the dev run; each tag has only one output folder
    for t in TAGS:
        src = PREV / "frozen" / f"{t}.json"
        dst = study_helpers.OUT / "frozen" / f"{t}.json"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        tm = PREV / "timing" / f"{t}.json"
        if tm.exists() and not (study_helpers.OUT / "timing" / f"{t}.json").exists():
            (study_helpers.OUT / "timing").mkdir(parents=True, exist_ok=True)
            shutil.copy(tm, study_helpers.OUT / "timing" / f"{t}.json")


def load(mode):
    return yes_no_scorer.Scorer(BIG, mode=mode)


frozen = {t: (json.load(open(study_helpers.OUT / "frozen" / f"{t}.json"))["settings"] if (study_helpers.OUT / "frozen" / f"{t}.json").exists() else None) for t in TAGS}
mode = frozen["f2_z8b"]["mode"] if frozen["f2_z8b"] else None
if mode:
    sc = load(mode)
else:
    try:
        mode, sc = "nf4", load("nf4")
    except Exception as e:
        study_helpers.log("nf4 failed, trying offload:", repr(e)[:200])
        torch.cuda.empty_cache()
        mode, sc = "off", load("off")
    study_helpers.log("mode", mode, "load s", round(sc.load_s))
for t, what in (("f2_z8b", "zero shot 8B over z2 K200"), ("f2_c8b", "cascade 8B over the zs06 top 50")):
    st = study_helpers.freeze(t, dict(model=BIG, mode=mode, K=200 if t == "f2_z8b" else 50, maxb=480, cands="gpu_z2s zs06 order",
                           builder="build_list.build SCAN 30 EARLIEST", note="memory mode chosen by whether nf4 loads, declared, no dev score used"))
    tags.row(t, "FIXED", f"{what}: {BIG} {st['mode']}; z2 list; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0",
             "dev timing covers scoring in this attempt only when a retry resumed from the cache")

for run in runs:
    if all(study_helpers.have(t, split, run) for t in TAGS):
        continue
    G = json.load(open(f"data/{run}/gpu_z2s.json"))
    qs, F, rn, P = study_helpers.builder(run)
    ctx = {p["id"]: p["ctx"] for p in P}
    qtext = {q["qid"]: f"{q['question']} (asked by {q['asker']})" for q in qs}
    base = {q["qid"]: study_helpers.ranked(G["cands"][q["qid"]], G["scores"][ZS06][q["qid"]]) for q in qs}
    cache = yes_no_scorer.Cache(RC / "z8b" / f"{run}.json")
    torch.cuda.reset_peak_memory_stats()
    t0 = sc.secs
    cache.fill(sc, {k: v[:50] for k, v in base.items()}, qtext, ctx)
    s1 = sc.secs - t0
    sub = {}
    for q in qs:
        b = base[q["qid"]]; top = [cache.d[q["qid"]][x] for x in b[:50]]; lo = min(top) - 1
        sub[q["qid"]] = study_helpers.build(F, q, rn, b, top + [lo - 1e-3 * i for i in range(len(b) - 50)])
    study_helpers.put("f2_c8b", split, run, sub, qs)
    study_helpers.timing("f2_c8b", run, s1, sc.peak(), mode=mode)
    cache.fill(sc, {k: v for k, v in base.items()}, qtext, ctx)
    sub = {q["qid"]: study_helpers.build(F, q, rn, base[q["qid"]], [cache.d[q["qid"]][x] for x in base[q["qid"]]]) for q in qs}
    study_helpers.put("f2_z8b", split, run, sub, qs)
    study_helpers.timing("f2_z8b", run, sc.secs - t0, sc.peak(), mode=mode)
    study_helpers.log("done", split, run, "pairs", sc.pairs, "secs", round(sc.secs - t0))
