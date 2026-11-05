"""python write_prefixes.py RUN [ROOT]: data/<run>/f3_pfx_<P1|P2|P3>.json = {piece_id: prefix + newline + ctx} for every piece of the run.
A style file that already exists is kept. Prefixes come from run_structure.py (no answer key)."""
import json, os, sys
from run_structure import load_run, STYLES

run = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else (os.environ.get("DATAROOT") or "data")
todo = [s for s in STYLES if not os.path.exists(f"{root}/{run}/f3_pfx_{s}.json")]
if todo:
    S = load_run(run, root)
    ctx = {json.loads(l)["id"]: json.loads(l)["ctx"] for l in open(f"{root}/{run}/pieces.jsonl", encoding="utf-8") if l.strip()}
    for st in todo:
        out = {p["id"]: S.prefix(p["id"], st) + "\n" + ctx[p["id"]] for p in S.P}
        tmp = f"{root}/{run}/f3_pfx_{st}.json.tmp"
        json.dump(out, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, f"{root}/{run}/f3_pfx_{st}.json")
        print(run, st, len(out), flush=True)
else:
    print(run, "all styles present", flush=True)
