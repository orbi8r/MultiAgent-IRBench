"""Trains a Qwen/Qwen3-Reranker-0.6B LoRA adapter for one epoch on the public rows, using train_reranker.py.
Reads the row files in study_helpers.PUBFILES (pub_00 to pub_14) and keeps every DIVth row (DIV 1 keeps all 4964).
Writes the adapter to $WORK/NAME and skips the run when that adapter already exists. Public rows only: no benchmark run is read.
    python train_ftpub.py NAME DIV SEED     -> $WORK/<NAME>"""
import os, subprocess, sys
from pathlib import Path
import study_helpers

name, div, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3]
work = Path(os.environ.get("WORKROOT", Path(os.environ["IRBENCH_SCRATCH"]) / "work"))
out = work / name
if (out / "adapter_model.safetensors").exists():
    study_helpers.log("adapter exists", out)
    sys.exit(0)
work.mkdir(parents=True, exist_ok=True)
rows = [l for fn in study_helpers.PUBFILES for l in open(study_helpers.BASE / "pubrows" / fn) if l.strip()]
rows = rows[::div][:int(os.environ.get("MAXROWS", 10 ** 9))]  # MAXROWS caps the row count for quick test runs only
rf = work / f"{name}.jsonl"
rf.write_text("".join(rows))
study_helpers.log("rows", len(rows), "of", div, "th")
env = dict(os.environ, SEED=seed)
subprocess.run([sys.executable, study_helpers.script("train_reranker.py"), "Qwen/Qwen3-Reranker-0.6B", str(out), "1", str(rf)], env=env, check=True)
rf.unlink()
