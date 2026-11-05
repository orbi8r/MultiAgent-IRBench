"""Key free. The original candidate features for OUR pool (fuse0 plus expansion, from the ftpub pool step) and the ftpub score of every pool piece.
The original scoring code (retrieve_with_scores in the original first stage module) is reused unchanged up to the ordering; the only difference is that the
candidates are the pool pieces, ordered by the original fused score `comb` (no dedup), instead of the original own top 120 after dedup.
    python pool_features.py OUT       writes $RL_STYLE_FEATS/pool_<run>.pkl for the dev and test runs (resumable per run)
RL_STYLE_FAKE_POOL=1 is for smoke tests only: the pool is the original first stage list and the placeholder ftpub score is the original BM25 column."""
import json, os, pickle, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

from rl_common import DEV, TEST, FEATS, limit, add_timing, peak_gb

OUT = Path(sys.argv[1])
BASE = Path(os.environ.get("IRBENCH_WORK", "."))
FAKE = os.environ.get("RL_STYLE_FAKE_POOL") == "1"
from method.pipeline import prepare_run
from method.config import TOP_K_CANDIDATES
from method import first_stage as FS

RRF_K, KM = FS.RRF_K, FS.KIND_MULTIPLIERS
FEATS.mkdir(parents=True, exist_ok=True)


def fused(ret, qtext, qtype):
    """the original steps 1 to 4 and move following and neighbour expansion, verbatim; returns (comb, b_scores, f_scores, sims, q_emb)"""
    b_scores = ret.bm25.score_dict(qtext)
    f_scores = ret.field_bm25.score_dict(qtext)
    q_emb = ret.dense.encode_query(qtext)
    sims = ret.dense.embeddings @ q_emb
    b_ranked = sorted(b_scores, key=lambda i: -b_scores[i]); b_rank = {i: r for r, i in enumerate(b_ranked)}
    f_ranked = sorted(f_scores, key=lambda i: -f_scores[i]); f_rank = {i: r for r, i in enumerate(f_ranked)}
    d_ranked = np.argsort(-sims).tolist(); d_rank = {i: r for r, i in enumerate(d_ranked)}
    pool = set(b_ranked[:600]) | set(f_ranked[:600]) | set(d_ranked[:600])
    rrf = defaultdict(float)
    for i in pool:
        if i in b_rank: rrf[i] += 1.0 / (RRF_K + b_rank[i])
        if i in f_rank: rrf[i] += 0.5 / (RRF_K + f_rank[i])
        if i in d_rank: rrf[i] += 1.0 / (RRF_K + d_rank[i])
    comb = {}
    for i, sc in rrf.items():
        k = ret.event_map.get(ret.block_event[ret.blocks[i]["id"]], {}).get("kind", "model")
        comb[i] = sc * KM.get(k, 1.0)
    if qtype == "location" or qtext.lower().startswith("where") or "location" in qtext.lower():
        for src, dst in ret.moves_map.items():
            if src in qtext or src.lower() in qtext.lower():
                base = dst.split("/")[-1]
                for i in range(len(ret.blocks)):
                    t = ret.blocks[i]["text"]
                    if dst in t or base in t:
                        comb[i] = max(comb.get(i, 0.0), 0.02)
    ev_max = defaultdict(float)
    for i, sc in comb.items():
        ev = ret.block_event[ret.blocks[i]["id"]]
        if sc > ev_max[ev]: ev_max[ev] = sc
    for ev, mx in ev_max.items():
        for i in ret.event_blocks[ev]:
            if mx * 0.5 > comb.get(i, 0.0): comb[i] = mx * 0.5
        if ev in ret.event_to_agent_pos:
            ag, pos = ret.event_to_agent_pos[ev]
            eids = ret.agent_events[ag]
            for dl in (-1, 1):
                if 0 <= pos + dl < len(eids):
                    for i in ret.event_blocks[eids[pos + dl]]:
                        if mx * 0.3 > comb.get(i, 0.0): comb[i] = mx * 0.3
    return comb, b_scores, f_scores, sims, q_emb


def feats_for(ret, qtext, qtype, sel, comb, b_scores, f_scores, sims):
    """the original retrieval_feats construction (10 columns) for the ordered index list `sel`"""
    N = len(sel)
    rf = np.zeros((N, 10), dtype=np.float32)
    nm = lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() - a.min() > 1e-9 else np.zeros_like(a)
    rf[:, 0] = nm(np.array([b_scores.get(i, 0.0) for i in sel], dtype=np.float32))
    rf[:, 1] = nm(np.array([f_scores.get(i, 0.0) for i in sel], dtype=np.float32))
    rf[:, 2] = np.array([sims[i] for i in sel], dtype=np.float32)
    rf[:, 3] = nm(np.array([comb.get(i, 0.0) for i in sel], dtype=np.float32))
    rf[:, 4] = 1.0 / (1.0 + np.arange(N, dtype=np.float32))
    rf[:, 5] = 1.0 / np.log2(2.0 + np.arange(N, dtype=np.float32))
    tok = np.array([max(ret.blocks[i].get("tokens", 64), 64) for i in sel], dtype=np.float32)
    rf[:, 6] = np.clip(tok / 1500.0, 0.0, 1.0)
    rf[:, 7] = (rf[:, 2] + rf[:, 3]) / np.maximum(np.log2(1.0 + tok / 64.0), 1.0)
    pt = [t.lower() for t in re.findall(r'[\w\-\./]+\.[a-zA-Z0-9]+', qtext) if len(t) > 3]
    if pt:
        rf[:, 8] = np.array([1.0 if any(p in ret.blocks[i]["text"].lower() for p in pt) else 0.0 for i in sel], dtype=np.float32)
    rf[:, 9] = np.array([1.0 if ret.event_map.get(ret.block_event[ret.blocks[i]["id"]], {}).get("kind", "") == "tool_call" else 0.0 for i in sel], dtype=np.float32)
    return rf


for r in DEV + [t for t in TEST if t not in DEV]:
    dst = FEATS / f"pool_{r}.pkl"
    if dst.exists():
        print(r, "resume: pool features exist", flush=True)
        continue
    t0 = time.time()
    rd = prepare_run(r)
    ret, bf = rd["retriever"], rd["block_features"]
    ret.minhash_threshold = None
    keep = set(limit([x["qid"] for x in rd["questions"]]))
    qs = [q for q in rd["questions"] if q["qid"] in keep]
    if FAKE:
        pools = None
    else:
        pools = json.load(open(BASE / "data" / r / "ftpub_pool.json"))
    idx_of = {b: i for i, b in enumerate(ret.block_ids)}
    items, miss, tot = {}, 0, 0
    for q in qs:
        qt = q.get("qtype", "")
        comb, bs, fsc, sims, qe = fused(ret, q["question"], qt)
        if FAKE:
            cand, _, _, rf0 = ret.retrieve_with_scores(q["question"], TOP_K_CANDIDATES, qtype=qt)
            pool = {b: float(v) for b, v in zip(cand, rf0[:, 0])}
        else:
            pool = pools.get(q["qid"], {})
        if not pool:
            continue
        ids = [b for b in pool if b in idx_of]
        miss += len(pool) - len(ids); tot += len(pool)
        order = sorted(range(len(ids)), key=lambda k: (-comb.get(idx_of[ids[k]], 0.0), k))
        ids = [ids[k] for k in order]
        sel = [idx_of[b] for b in ids]
        rf = feats_for(ret, q["question"], qt, sel, comb, bs, fsc, sims)
        sf = np.stack([bf.get(b, np.zeros(15, dtype=np.float32)) for b in ids])
        items[q["qid"]] = dict(block_ids=ids, retrieval_feats=rf, struct_feats=sf, question=q["question"], qtype=qt,
                               ftpub=np.array([pool[b] for b in ids], dtype=np.float32))
    # id space check: about 2 percent of our pieces are events with no block, those are dropped; a wrong id space would miss nearly all
    assert tot == 0 or miss / tot < 0.2, f"{r}: {miss} of {tot} pool ids are not blocks of the run"
    pickle.dump(items, open(dst, "wb"))
    add_timing(OUT, "pl_hybrid_feats", r, time.time() - t0, peak_gb())
    print(r, "done", len(items), "questions, pool ids missing", miss, "of", tot, round(time.time() - t0), "s", flush=True)
