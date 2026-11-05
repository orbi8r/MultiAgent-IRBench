"""Fast pipeline configs over every run (timed_run.py, ftpub reader, ds_bgeb student, lexical fuse0 first stage), one process per run.
    python timed_run_configs.py CFG[,CFG...] [TAGPREFIX]     tag = <TAGPREFIX or f2_fast_><CFG>, subs plus timing (total without model load, peak GPU memory)
timed_run.py output lists stay under $WORK and are deleted per run; only the subs and timing json are kept."""
import json, os, subprocess, sys
from pathlib import Path
import study_helpers, tags

cfgs = sys.argv[1].split(",")
pref = sys.argv[2] if len(sys.argv) > 2 else "f2_fast_"
work = Path(os.environ.get("WORKROOT", Path(os.environ["IRBENCH_SCRATCH"]) / "work")) / "fast"
work.mkdir(parents=True, exist_ok=True)
frozen = {}
for c in cfgs:
    t = pref + c
    frozen[t] = study_helpers.freeze(t, dict(config=c, adapter="models/ftpub", student="models/ds_bgeb", first="fuse0 (lexical, RRF of seven lists)",
                                  script="timed_run.py", builder="build_list.build SCAN 30 EARLIEST", note="timing is total without model load"))
    tags.row(t, "FIXED", f"timed_run.py {c}, ftpub reader, ds_bgeb student where the config starts with S; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0",
             "timing is lexical first stage plus reader plus list builder without model load")
for split, run in study_helpers.plan():
    todo = [c for c in cfgs if not study_helpers.have(pref + c, split, run)]
    if not todo:
        continue
    out = work / f"{run}_{split}.json"
    env = dict(os.environ, NPROC=os.environ.get("NPROC", "8"), OMP_NUM_THREADS="4", STUDENT=str(study_helpers.BASE / "models" / "ds_bgeb"))
    subprocess.run([sys.executable, study_helpers.script("timed_run.py"), run, str(study_helpers.BASE / "models" / "ftpub"), ",".join(cfgs), str(out)], env=env, check=True)
    R = json.load(open(out))
    qs = study_helpers.questions(run)
    for c in todo:
        study_helpers.put(pref + c, split, run, R["lists"][c], qs)
        study_helpers.timing(pref + c, run, R["times"][f"{c}_total_excl_model_load"], R["times"].get("peak_gb", 0.0), pairs=R["times"][f"{c}_pairs"])
    out.unlink()
    study_helpers.log("done", split, run, todo)
