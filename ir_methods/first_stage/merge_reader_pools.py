"""Merge the two ftpub pool rounds into data/<run>/ftpub_pool.json = {qid: {piece_id: log p(yes)}} and check that the
fuse0 and base top 200 are all scored (the pool lists are data/<run>/pool1.json, pool2.json, f1.json)."""
import json, sys
run = sys.argv[1]
D = f"data/{run}"
out = {}
for f in ("pubsub1", "pubsub2"):
    for qid, d in json.load(open(f"{D}/{f}.json")).items():
        out.setdefault(qid, {}).update(d)
f1 = json.load(open(f"{D}/f1.json"))
miss = sum(1 for l in f1.values() for qid, ids in l.items() for x in ids if x not in out.get(qid, {}))
assert miss == 0, f"{run}: {miss} first stage pieces not scored"
json.dump(out, open(f"{D}/ftpub_pool.json", "w"))
print(run, len(out), "questions", sum(len(v) for v in out.values()), "scored pairs")
