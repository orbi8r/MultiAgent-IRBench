"""Key free stage (P0). No answer key is read. Per run and dedup setting it writes the first stage list (pl_fs) and the
candidate features the original policies rank, into FEATS (persistent, resumable per run).
    python features.py OUT"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np

from rl_common import DEV, TEST, DEDUPS, FEATS, limit, add_timing, peak_gb

OUT = Path(sys.argv[1])
from method.pipeline import prepare_run
from method.config import TOP_K_CANDIDATES

TH = {"exact": None, "mh85": 0.85}  # exact SHA was the original headline setting, MinHash 0.85 the repo default; both are rows
FEATS.mkdir(parents=True, exist_ok=True)

for r in DEV + [t for t in TEST if t not in DEV]:
    done = [FEATS / f"fs_{r}_{d}.json" for d in DEDUPS] + [FEATS / f"items_{r}_{d}.pkl" for d in DEDUPS]
    if all(p.exists() for p in done):
        print(r, "resume: features exist", flush=True)
        continue
    t0 = time.time()
    rd = prepare_run(r)
    ret, bf = rd["retriever"], rd["block_features"]
    keep = set(limit([x["qid"] for x in rd["questions"]]))
    qs = [q for q in rd["questions"] if q["qid"] in keep]
    for d in DEDUPS:
        ret.minhash_threshold = TH[d]
        items, fs = {}, {}
        for q in qs:
            qt = q.get("qtype", "")
            cand, qe, ce, rf = ret.retrieve_with_scores(q["question"], TOP_K_CANDIDATES, qtype=qt)
            fs[q["qid"]] = ret.retrieve(q["question"], TOP_K_CANDIDATES, qtype=qt)
            if not cand:
                continue
            sf = np.stack([bf.get(b, np.zeros(15, dtype=np.float32)) for b in cand])
            items[q["qid"]] = dict(block_ids=list(cand), retrieval_feats=np.asarray(rf, dtype=np.float32), struct_feats=sf,
                                   question=q["question"], qtype=qt, query_emb=qe, block_embs=ce)
        pickle.dump(items, open(FEATS / f"items_{r}_{d}.pkl", "wb"))
        json.dump(fs, open(FEATS / f"fs_{r}_{d}.json", "w"))
    secs = time.time() - t0
    add_timing(OUT, "pl_feats", r, secs, peak_gb())
    print(r, "done", round(secs, 1), "s", flush=True)
