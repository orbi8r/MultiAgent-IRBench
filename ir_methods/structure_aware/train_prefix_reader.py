"""Trains a reader LoRA on the prefixed public rows of one style (batch 1, pub_00 to pub_14, 4964 rows, 1 epoch, via train_reranker.py), with the same recipe as the plain reranker fine tune.
    python train_prefix_reader.py NAME STYLE SEED|none [BASE_MODEL]    -> $IRBENCH_WORK/adapters3/NAME (skipped when it already exists), default base Qwen3 Reranker 0.6B
Rows come from $IRBENCH_WORK/pubrows3/<STYLE>/pub_NN.jsonl (prefix_training_rows.py). Rows keep file order, DIV 1 and SEED is passed to train_reranker.py through the environment.
Env STEPCKPT (steps between checkpoints, 0 off) keeps $IRBENCH_WORK/tmp/ck3/NAME.pt so a killed attempt resumes. MAXROWS only for smokes."""
import os, subprocess, sys
from pathlib import Path
import study_helpers

name, style, seed = sys.argv[1:4]
base = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen3-Reranker-0.6B"
out = study_helpers.BASE / "adapters3" / name
if (out / "adapter_model.safetensors").exists():
    study_helpers.log("adapter exists", out)
    sys.exit(0)
work = Path(os.environ.get("WORKROOT", Path(os.environ["IRBENCH_SCRATCH"]) / "work"))
work.mkdir(parents=True, exist_ok=True); out.parent.mkdir(parents=True, exist_ok=True)
root = Path(os.environ.get("ROWS3", study_helpers.BASE / "pubrows3")) / style
files = [fn for fn in study_helpers.PUBFILES if (root / fn).exists() or "MAXROWS" not in os.environ]  # a smoke may hold only pub_00
rows = [l for fn in files for l in open(root / fn) if l.strip()]
rows = rows[:int(os.environ.get("MAXROWS", 10 ** 9))]
rf = work / f"{name}.jsonl"
rf.write_text("".join(rows))
study_helpers.log("rows", len(rows), "style", style, "seed", seed)
env = dict(os.environ)
if seed != "none":  # none keeps the unseeded recipe of ftpub4b (shuffle seed 0)
    env["SEED"] = seed
else:
    env.pop("SEED", None)
ck = int(os.environ.get("STEPCKPT", 0))
if ck:
    (study_helpers.BASE / "tmp" / "ck3").mkdir(parents=True, exist_ok=True)
    env.update(CKPT=str(study_helpers.BASE / "tmp" / "ck3" / f"{name}.pt"), CKEVERY=str(ck))
subprocess.run([sys.executable, study_helpers.script("train_reranker.py"), base, str(out), "1", str(rf)], env=env, check=True)
rf.unlink()
