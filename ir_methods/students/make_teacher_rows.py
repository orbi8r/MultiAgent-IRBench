"""Attaches the public ensemble teacher to the public rows. The teacher is the z scored 4B reader score plus 0.5 times the z scored 0.6B reader score, both scored on public rows.
    python make_teacher_rows.py OUT.jsonl  (reads data/pub/*.jsonl and teach/t4b_*.jsonl, teach/t06_*.jsonl)"""
import glob, json, statistics as st, sys
from collections import defaultdict

T = defaultdict(dict)
for tag, w in (("t4b", 1.0), ("t06", 0.5)):
    for f in glob.glob(f"teach/{tag}_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            s = r["scores"]; m, sd = st.mean(s), st.pstdev(s) or 1.0
            T[(r["file"], r["line"])][tag] = [w * (x - m) / sd for x in s]
n = 0
with open(sys.argv[1], "w") as fo:
    for f in sorted(glob.glob("data/pub/*.jsonl")):
        for ln, l in enumerate(open(f)):
            t = T.get((f, ln), {})
            if len(t) < 2:
                continue
            r = json.loads(l)
            k = min(len(t["t4b"]), len(t["t06"]))
            r["negs"] = r["negs"][:k - 1]
            r["teacher"] = [a + b for a, b in zip(t["t4b"][:k], t["t06"][:k])]
            fo.write(json.dumps(r) + "\n"); n += 1
print("rows with teacher", n)
