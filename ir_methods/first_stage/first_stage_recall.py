"""First stage variants, recall only (no reranker): share of non location atoms with a credited piece in the top K, and
with a near cheapest credited piece (cost <= 1.5 x cheapest + 50) in the top K. Keys used only to measure.
    python first_stage_recall.py [variant ...]"""
import json, sys
from collections import defaultdict, Counter
import os
os.environ.setdefault("IRBENCH_KEYS", "1")  # diagnostic: scores against the answer key
from run_loader import load, S, RUNS
from contextual_pieces import toks
from bm25_tools import BM, rank, rrf
from novelty import novelty

KS = (10, 25, 50, 100, 200, 400)


def lists_for(name, P, qs, variants):
    nov = novelty(P)
    ids = [p["id"] for p in P]
    tc = [toks(p["ctx"]) for p in P]
    tp = [toks(p["text"]) for p in P]
    full_c, full_p = BM(tc), BM(tp)
    keep = [i for i, p in enumerate(P) if nov[p["id"]] > 0.2]
    nov_c, nov_p = BM([tc[i] for i in keep]), BM([tp[i] for i in keep])
    out = defaultdict(dict)
    for q in qs:
        qt = toks(q["question"])
        base = rrf(rank(full_c.scores(qt)), rank(full_p.scores(qt)))
        if "base" in variants:
            out["base"][q["qid"]] = [ids[i] for i in base]
        if "nov" in variants:
            l = rrf(rank(nov_c.scores(qt)), rank(nov_p.scores(qt)))
            out["nov"][q["qid"]] = [ids[keep[i]] for i in l]
        if "asker" in variants:  # asker's own view ranked inside the full index, fused with the global list
            sc = full_c.scores(qt)
            mine = [i for i in rank(sc, 10 ** 6) if P[i]["agent"] == q["asker"]][:2000]
            out["asker"][q["qid"]] = [ids[i] for i in rrf(base, mine)]
    return out


if __name__ == "__main__":
    variants = sys.argv[1:] or ["base", "nov", "asker"]
    R = defaultdict(Counter)
    for name in RUNS:
        d, run, key, qs, blocks, events = load(name)
        P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
        L = lists_for(name, P, qs, variants)
        tok = run["tokens"]
        for q in qs:
            if q["qtype"] == "location":
                continue
            for a in S.credits(key[q["qid"]], run):
                cheap = min(tok[x] for x in a)
                near = {x for x in a if tok[x] <= 1.5 * cheap + 50}
                for v in variants:
                    pos = {x: i for i, x in enumerate(L[v][q["qid"]])}
                    r = min((pos[x] for x in a if x in pos), default=10 ** 9)
                    rn = min((pos[x] for x in near if x in pos), default=10 ** 9)
                    R[v]["n"] += 1
                    for k in KS:
                        R[v][f"any@{k}"] += r < k; R[v][f"near@{k}"] += rn < k
    for v in variants:
        n = R[v]["n"]
        print(v, "any", " ".join(f"@{k}={R[v][f'any@{k}']/n:.3f}" for k in KS), "| near", " ".join(f"@{k}={R[v][f'near@{k}']/n:.3f}" for k in KS), f"(n={n})")
