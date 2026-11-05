"""Scores the frozen z2 candidates (data/<run>/gpu_z2.json, written by the prep step) with rerankers, then writes dev and test subs.
    python score_z2.py GTAG "tag|model|mode" ...      mode build (list builder), raw (reader order) or first (cands order, model -)
Reader scores go to data/<run>/gpu_<GTAG>.json (gpu_score.py), with one .ok_<GTAG> marker per run. Env MAXB passes through.
Every tag is frozen first (declared settings), then dev subs, then test subs. Timing = reader scoring seconds without model load."""
import json, os, subprocess, sys
import study_helpers, tags

gtag, specs = sys.argv[1], [s.split("|") for s in sys.argv[2:]]
models = list(dict.fromkeys(m for _, m, _ in specs if m != "-"))
maxb = int(os.environ.get("MAXB", 480))
extra = json.loads(os.environ.get("FROZEN_EXTRA", "{}"))
frozen = {}
for tag, model, mode in specs:
    frozen[tag] = study_helpers.freeze(tag, dict(model=model, mode=mode, K=200, maxb=maxb, cands="gpu_z2 (RRF of ctx BM25, plain BM25, Qwen3 embedding 0.6B dense)",
                                      builder="build_list.build SCAN 30 EARLIEST", **extra))
    st = frozen[tag]
    tags.row(tag, "FIXED", f"{st['model']} {st['mode']} K200 maxb {st['maxb']}; z2 list; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0")
specs = [(t, frozen[t]["model"], frozen[t]["mode"]) for t, _, _ in specs]  # test path reads the frozen settings
for split, run in study_helpers.plan():
    todo = [s for s in specs if not study_helpers.have(s[0], split, run)]
    if not todo:
        continue
    ok = f"data/{run}/.ok_{gtag}"
    if models and not os.path.exists(ok):
        env = dict(os.environ, CANDS_FROM=f"data/{run}/gpu_z2.json")
        subprocess.run([sys.executable, study_helpers.script("gpu_score.py"), run, gtag, "none", ",".join(models), "200"], env=env, check=True)
        open(ok, "w").close()
    for tag, model, mode in todo:
        qs, sub, G = study_helpers.sub_from_gpu(run, gtag if model != "-" else "z2", model, mode)
        study_helpers.put(tag, split, run, sub, qs)
        if model != "-":
            t = G["times"]
            study_helpers.timing(tag, run, t[f"score_{model}"], t[f"peak_gb_{model}"], load=t[f"load_{model}"], pairs=len(qs) * 200)
    study_helpers.log("done", split, run, [s[0] for s in todo])
