"""Builds the c4b cascade subs (f2_c4b) and the timing only tags t2_c4b, t2_ds_bgeb and t2_ftpub_k200 for every run.
    python cascade_timing.py c4b       cascade.py (the 4B reranker rereads the top 50 of the 0.6B reranker's ranking on z2) -> f2_c4b subs; needs data/<run>/gpu_z2s.json
    python cascade_timing.py t2        timing tags from measured parts: lexical seconds + dense seconds + reader seconds (no model load); t2_ds_bgeb needs gpu_z2ds.json
    python cascade_timing.py fast      t2_ftpub_k200 = timed_run.py K200 on fuse0 with ftpub, total without model load
Timing only tags have a frozen file and a row but no sub."""
import json, os, subprocess, sys
from pathlib import Path
import study_helpers, tags

mode = sys.argv[1]
ZS06 = "Qwen/Qwen3-Reranker-0.6B"
M4 = os.environ.get("M4", "Qwen/Qwen3-Reranker-4B")
work = Path(os.environ.get("WORKROOT", Path(os.environ["IRBENCH_SCRATCH"]) / "work")) / "tim"
work.mkdir(parents=True, exist_ok=True)


def lex(run):
    return float(open(f"data/{run}/lex_secs.txt").read())


if mode == "c4b":
    t = "f2_c4b"
    st = study_helpers.freeze(t, dict(base=ZS06, big=M4, N=50, gfile="gpu_z2s", script="cascade.py", builder="build_list.build SCAN 30 EARLIEST"))
    tags.row(t, "FIXED", f"cascade.py: {M4} rereads the {ZS06} top 50 of the z2 list; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0",
             "timing is the 4B scoring of the top 50 only")
    for split, run in study_helpers.plan():
        if study_helpers.have(t, split, run):
            continue
        if not os.path.exists(f"data/{run}/gpu_f2c4b.json"):
            subprocess.run([sys.executable, study_helpers.script("cascade.py"), run, "z2s", ZS06, M4, "50", "f2c4b"], check=True)
        G = json.load(open(f"data/{run}/gpu_f2c4b.json"))
        qs, F, rn, P = study_helpers.builder(run)
        sub = {q["qid"]: study_helpers.build(F, q, rn, G["cands"][q["qid"]], G["scores"][M4][q["qid"]]) for q in qs}
        study_helpers.put(t, split, run, sub, qs)
        study_helpers.timing(t, run, G["times"]["score"], 0.0, load=G["times"]["load"])
        study_helpers.log("done", split, run)
elif mode == "t2":
    for t, why in (("t2_c4b", "lexical, dense, zs06 over K200 and the 4B over the top 50, scoring seconds without model load"),
                   ("t2_ds_bgeb", "lexical, dense and the ds_bgeb student over K200, scoring seconds without model load")):
        study_helpers.freeze(t, dict(parts=why, note="sum of measured parts per run, timing only"))
        tags.row(t, "PROBE", why, "timing only, no sub")
    for split, run in study_helpers.plan():
        if split == "test" and run in study_helpers.DEVR:
            continue  # smoke runs only: the same run is both splits
        z = json.load(open(f"data/{run}/gpu_z2.json"))["times"]
        dense = z.get("dense", 0.0) + 0.0
        a = json.load(open(f"data/{run}/gpu_z2s.json"))["times"]
        c = json.load(open(f"data/{run}/gpu_f2c4b.json"))["times"]
        s = json.load(open(f"data/{run}/gpu_z2ds.json"))["times"]
        base = lex(run) + dense
        study_helpers.timing("t2_c4b", run, base + a[f"score_{ZS06}"] + c["score"], a[f"peak_gb_{ZS06}"], lexical=lex(run), dense=dense, zs06=a[f"score_{ZS06}"], big=c["score"])
        study_helpers.timing("t2_ds_bgeb", run, base + s["score_models/ds_bgeb"], s["peak_gb_models/ds_bgeb"], lexical=lex(run), dense=dense, student=s["score_models/ds_bgeb"])
else:
    t = "t2_ftpub_k200"
    study_helpers.freeze(t, dict(config="K200", adapter="models/ftpub", first="fuse0", script="timed_run.py", note="total without model load, timing only"))
    tags.row(t, "PROBE", "timed_run.py K200 on fuse0 with ftpub, total seconds without model load", "timing only, no sub")
    done = set()
    for split, run in study_helpers.plan():
        if run in done:
            continue
        done.add(run)
        out = work / f"{run}.json"
        env = dict(os.environ, NPROC=os.environ.get("NPROC", "8"), OMP_NUM_THREADS="4")
        env.pop("STUDENT", None)
        subprocess.run([sys.executable, study_helpers.script("timed_run.py"), run, str(study_helpers.BASE / "models" / "ftpub"), "K200", str(out)], env=env, check=True)
        R = json.load(open(out))["times"]
        study_helpers.timing(t, run, R["K200_total_excl_model_load"], R.get("peak_gb", 0.0), pairs=R["K200_pairs"])
        out.unlink()
