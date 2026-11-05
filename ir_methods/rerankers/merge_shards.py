"""Merges the shard files of every run in RUNS into one file per run.
    python merge_shards.py PREFIX  (data/<run>/PREFIX.sh*.json -> data/<run>/PREFIX.json)"""
import glob, json, sys
from run_loader import RUNS
for name in RUNS:
    out = {}
    fs = sorted(glob.glob(f"data/{name}/{sys.argv[1]}.sh*.json"))
    for f in fs:
        for qid, d in json.load(open(f)).items():
            out.setdefault(qid, {}).update(d)
    json.dump(out, open(f"data/{name}/{sys.argv[1]}.json", "w"))
    print(name, len(fs), "shards", len(out), "questions", sum(len(v) for v in out.values()), "pairs")
