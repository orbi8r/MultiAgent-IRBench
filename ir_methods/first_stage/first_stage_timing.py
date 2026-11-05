"""Writes the first stage timing tag t2_z2_first, which is the lexical seconds plus the dense seconds for each run. Inputs are
data/<run>/lex_secs.txt and data/<run>/gpu_z2.json, the runs come from the ALL environment variable, and the output is
timing/t2_z2_first.json under IRBENCH_OUT."""
import json, os
from pathlib import Path
BASE, OUT = Path(os.environ["IRBENCH_WORK"]), Path(os.environ["IRBENCH_OUT"])
lines, tim = [], {}
for run in os.environ["ALL"].split():  # the runs this task built (RUNSEL narrows it in smokes)
    D = Path("data") / run
    g = json.load(open(D / "gpu_z2.json"))["times"]
    lex = float(open(D / "lex_secs.txt").read())
    tim[run] = dict(secs=lex + g.get("dense", 0.0), peak_gb=0.0, parts=dict(lexical=lex, dense=g.get("dense", 0.0)))
(OUT / "timing").mkdir(exist_ok=True)
json.dump(tim, open(OUT / "timing" / "t2_z2_first.json", "w"), indent=1)
