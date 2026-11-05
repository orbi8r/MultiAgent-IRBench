"""full3 scorer of one reader over the frozen z2 K200 candidates with prefixed candidate text (CTXFILE), dev and or test subs.
    python score_prefix_reader.py --tag T --model M --style P1|P2|P3|none --splits dev|test|both [--freeze] [--kind K] [--train TEXT]
Same candidates (data/<run>/gpu_z2.json), MAXB and prompt as the z2 reranker run: only the document text changes, ids stay (data/<run>/f3_pfx_<style>.json).
Reader scores go to data/<run>/gpu_<tag>.json (marker .ok_<tag>), subs through study_helpers.build the same way as the z2 run. A test split needs the frozen file:
it is written here with --freeze, or earlier by the task (pick_prefix_style.py), and read back so the test path uses the frozen settings."""
import argparse, json, os, subprocess, sys
import study_helpers

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--style", required=True)
ap.add_argument("--splits", default="both")
ap.add_argument("--freeze", action="store_true")
ap.add_argument("--norow", action="store_true")  # the dev run already wrote this tag's row
ap.add_argument("--kind", default="FIXED")
ap.add_argument("--train", default="")
ap.add_argument("--section", default="9.1 full3 prefixed reader over z2 K200")
ap.add_argument("--method", default="")
ap.add_argument("--note", default="")
ap.add_argument("--evidence", default="declared, no choice")
a = ap.parse_args()
maxb = int(os.environ.get("MAXB", 480))
flat = a.style == "none"
settings = dict(model=a.model, mode="build", K=200, maxb=maxb, cands="gpu_z2 (RRF of ctx BM25, plain BM25, Qwen3 embedding 0.6B dense)",
                builder="build_list.build SCAN 30 EARLIEST", prefix_style=a.style, ctxfile="none" if flat else "data/<run>/f3_pfx_<style>.json")
if a.train:
    settings["train"] = a.train
wants = ("dev", "test") if a.splits == "both" else (a.splits,)
if a.freeze or "test" in wants:
    settings = study_helpers.freeze(a.tag, settings, json.loads(a.evidence) if a.evidence.startswith(("{", "[")) else a.evidence)
model = settings["model"]
if not a.norow:
    study_helpers.row(a.tag, a.section, a.method or f"{model.split('=')[-1]} on z2 K200, prefix {a.style}, list builder", a.kind,
           f"{model} build K200 maxb {maxb}; z2 list; prefix {a.style}; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0", "", a.note)
for split, run in study_helpers.plan():
    if split not in wants or study_helpers.have(a.tag, split, run):
        continue
    ok = f"data/{run}/.ok_{a.tag}"
    if not os.path.exists(ok):
        env = dict(os.environ, CANDS_FROM=f"data/{run}/gpu_z2.json")
        env.pop("CTXFILE", None)
        if not flat:
            env["CTXFILE"] = f"data/{run}/f3_pfx_{a.style}.json"
        subprocess.run([sys.executable, study_helpers.script("gpu_score.py"), run, a.tag, "none", model, "200"], env=env, check=True)
        open(ok, "w").close()
    qs, sub, G = study_helpers.sub_from_gpu(run, a.tag, model, "build")
    study_helpers.put(a.tag, split, run, sub, qs)
    t = G["times"]
    study_helpers.timing(a.tag, run, t[f"score_{model}"], t[f"peak_gb_{model}"], load=t[f"load_{model}"], pairs=len(qs) * 200)
    study_helpers.log("done", a.tag, split, run)
