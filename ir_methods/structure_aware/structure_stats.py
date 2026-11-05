"""python structure_stats.py OUTFILE RUN...: structure stats per run (nodes per type) into OUTFILE, plus prefix token lengths per style when the tokenizer is cached."""
import json, os, sys
from run_structure import load_run, STYLES

out, runs = sys.argv[1], sys.argv[2:]
res = json.load(open(out)) if os.path.exists(out) else {}
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
except Exception as e:
    tok = None
    print("no tokenizer:", str(e)[:100], flush=True)
for r in runs:
    if r in res:
        continue
    S = load_run(r)
    st = S.stats()
    if tok:
        ids = [p["id"] for p in S.P][::10]
        for s in STYLES:
            n = sorted(len(tok.encode(S.prefix(i, s), add_special_tokens=False)) for i in ids)
            st[f"prefix_tokens_{s}"] = dict(mean=round(sum(n) / len(n), 1), p99=n[int(0.99 * len(n))], max=n[-1])
    res[r] = st
    json.dump(res, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
    print(r, json.dumps(st)[:300], flush=True)
