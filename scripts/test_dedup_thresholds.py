"""
Test candidate recall and BAR metrics with MinHash deduplication at various thresholds.
Evaluates on training runs: library_maintainers, literature_review, online_shop.
"""
import sys
import json
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from method.config import RUNS_DIR, TOP_K_CANDIDATES
from method.graph_builder import load_run, build_graph
from method.features import extract_all_block_features, normalise_features
from method.first_stage import FirstStageRetriever
from method.reranker import RLReranker, SubmodularDirectPolicy
from score import score_query, score_query_equal, credits as compute_credits

M = 128
PRIME = 2147483647
rng = np.random.RandomState(42)
A = rng.randint(1, PRIME, size=(1, M), dtype=np.int64)
B = rng.randint(0, PRIME, size=(1, M), dtype=np.int64)

def compute_minhash_signature(text: str) -> np.ndarray:
    words = text.strip().lower().split()
    if not words:
        return np.zeros(M, dtype=np.int64)
    shingles = set(words)
    for i in range(len(words) - 1):
        shingles.add(f"{words[i]} {words[i+1]}")
    hashes = np.array([zlib.crc32(s.encode("utf-8")) for s in shingles], dtype=np.int64).reshape(-1, 1)
    return ((hashes * A + B) % PRIME).min(axis=0)

def compute_all_signatures(blocks):
    sigs = np.empty((len(blocks), M), dtype=np.int64)
    for i, b in enumerate(blocks):
        sigs[i] = compute_minhash_signature(b["text"])
    return sigs

def retrieve_candidates(retriever, all_sigs, question, qtype, top_k, threshold=None):
    """
    If threshold is None: exact SHA deduplication.
    If threshold is float in (0, 1]: MinHash deduplication with exact SHA fast-path.
    """
    from collections import defaultdict
    from method.config import RRF_K
    from method.first_stage import KIND_MULTIPLIERS

    qtext = question

    # 1. Block BM25
    b_scores = retriever.bm25.score_dict(qtext)

    # 2. Field BM25
    f_scores = retriever.field_bm25.score_dict(qtext)

    # 3. Dense cosine
    q_emb = retriever.dense.encode_query(qtext)
    sims = retriever.dense.embeddings @ q_emb  # (M,)

    # 4. RRF fusion
    b_ranked = sorted(b_scores, key=lambda i: -b_scores[i])
    b_rank = {idx: r for r, idx in enumerate(b_ranked)}
    f_ranked = sorted(f_scores, key=lambda i: -f_scores[i])
    f_rank = {idx: r for r, idx in enumerate(f_ranked)}
    d_ranked = np.argsort(-sims).tolist()
    d_rank = {idx: r for r, idx in enumerate(d_ranked)}

    pool = set(b_ranked[:600]) | set(f_ranked[:600]) | set(d_ranked[:600])
    rrf = defaultdict(float)
    for idx in pool:
        if idx in b_rank:
            rrf[idx] += 1.0 / (RRF_K + b_rank[idx])
        if idx in f_rank:
            rrf[idx] += 0.5 / (RRF_K + f_rank[idx])
        if idx in d_rank:
            rrf[idx] += 1.0 / (RRF_K + d_rank[idx])

    # Kind multipliers
    comb = {}
    for idx, sc in rrf.items():
        eid = retriever.block_event[retriever.blocks[idx]["id"]]
        k = retriever.event_map.get(eid, {}).get("kind", "model")
        comb[idx] = sc * KIND_MULTIPLIERS.get(k, 1.0)

    # Move following
    is_loc = qtype == "location" or qtext.lower().startswith("where") or "location" in qtext.lower()
    if is_loc:
        for src, dst in retriever.moves_map.items():
            if src in qtext or src.lower() in qtext.lower():
                dst_base = dst.split("/")[-1]
                for idx in range(len(retriever.blocks)):
                    btext = retriever.blocks[idx]["text"]
                    if dst in btext or dst_base in btext:
                        comb[idx] = max(comb.get(idx, 0.0), 0.02)

    # Neighbour expansion
    ev_max = defaultdict(float)
    for idx, sc in comb.items():
        ev = retriever.block_event[retriever.blocks[idx]["id"]]
        if sc > ev_max[ev]:
            ev_max[ev] = sc
    for ev, mx in ev_max.items():
        sh_same = mx * 0.5
        for idx in retriever.event_blocks[ev]:
            if sh_same > comb.get(idx, 0.0):
                comb[idx] = sh_same
        if ev in retriever.event_to_agent_pos:
            ag, pos = retriever.event_to_agent_pos[ev]
            eids = retriever.agent_events[ag]
            sh_adj = mx * 0.3
            for delta in [-1, 1]:
                if 0 <= pos + delta < len(eids):
                    adj_ev = eids[pos + delta]
                    for idx in retriever.event_blocks[adj_ev]:
                        if sh_adj > comb.get(idx, 0.0):
                            comb[idx] = sh_adj

    best = sorted(comb, key=lambda idx: -comb[idx])

    selected_indices = []
    seen_hashes = set()

    if threshold is None:
        for idx in best:
            h = retriever.block_hashes[idx]
            if h not in seen_hashes:
                seen_hashes.add(h)
                selected_indices.append(idx)
                if len(selected_indices) >= top_k:
                    break
    else:
        threshold_matches = int(threshold * M)
        selected_sigs = np.empty((top_k, M), dtype=np.int64)
        for idx in best:
            h = retriever.block_hashes[idx]
            if h in seen_hashes:
                continue
            cand_sig = all_sigs[idx]
            n_sel = len(selected_indices)
            if n_sel > 0:
                matches = (selected_sigs[:n_sel] == cand_sig).sum(axis=1)
                if np.any(matches >= threshold_matches):
                    continue
            seen_hashes.add(h)
            selected_sigs[n_sel] = cand_sig
            selected_indices.append(idx)
            if len(selected_indices) >= top_k:
                break

    block_ids = [retriever.block_ids[idx] for idx in selected_indices]
    N = len(selected_indices)
    block_embs = retriever.dense.embeddings[selected_indices] if selected_indices else np.zeros((0, 384), dtype=np.float32)

    # Retrieval feats
    retrieval_feats = np.zeros((N, 10), dtype=np.float32)
    if N > 0:
        raw_bm25 = np.array([b_scores.get(idx, 0.0) for idx in selected_indices], dtype=np.float32)
        raw_field = np.array([f_scores.get(idx, 0.0) for idx in selected_indices], dtype=np.float32)
        raw_dense = np.array([sims[idx] for idx in selected_indices], dtype=np.float32)
        raw_rrf = np.array([comb.get(idx, 0.0) for idx in selected_indices], dtype=np.float32)

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-9:
                return (arr - mn) / (mx - mn)
            return np.zeros_like(arr)

        retrieval_feats[:, 0] = _norm(raw_bm25)
        retrieval_feats[:, 1] = _norm(raw_field)
        retrieval_feats[:, 2] = np.clip((raw_dense + 1.0) / 2.0, 0.0, 1.0)
        retrieval_feats[:, 3] = _norm(raw_rrf)
        retrieval_feats[:, 4] = retrieval_feats[:, 0] * retrieval_feats[:, 2]
        ranks = np.arange(1, N + 1, dtype=np.float32)
        retrieval_feats[:, 5] = 1.0 / np.log2(ranks + 1.0)
        costs = np.array([max(retriever.tokens_map.get(b, 64), 64) for b in block_ids], dtype=np.float32)
        retrieval_feats[:, 6] = np.clip(costs / 500.0, 0.0, 2.0)
        retrieval_feats[:, 7] = (retrieval_feats[:, 0] + retrieval_feats[:, 2]) / (costs ** 0.01)
        q_lower = question.lower()
        for i, b in enumerate(block_ids):
            txt = retriever.blocks[selected_indices[i]]["text"].lower()
            ev_id = retriever.block_event.get(b, "")
            ev = retriever.event_map.get(ev_id, {})
            ev_text = ev.get("text", "").lower()
            retrieval_feats[i, 8] = 1.0 if any(
                term in txt or term in ev_text
                for term in ["path", "file", "/", ".py", ".md", ".json", ".txt"]
                if term in q_lower
            ) else 0.0
            retrieval_feats[i, 9] = 1.0 if ev.get("kind") == "tool_call" else 0.0

    return block_ids, q_emb, block_embs, retrieval_feats


if __name__ == "__main__":
    from score import load_run as score_load_run
    run_name = "library_maintainers"
    print(f"Loading {run_name} once...")
    run_path = RUNS_DIR / run_name
    events, blocks, questions = load_run(run_path)
    run_obj = score_load_run(str(run_path))
    with open(run_path / "answer_key.json") as f:
        ak = json.load(f)

    tokens_map = {}
    for e in events:
        tokens_map[e["id"]] = e["tokens"]
    for b in blocks:
        tokens_map[b["id"]] = b["tokens"]

    graph = build_graph(events, blocks)
    feats = normalise_features(extract_all_block_features(blocks, events, graph))
    retriever = FirstStageRetriever(blocks, feats, events=events, use_dense=True)
    all_sigs = compute_all_signatures(blocks)

    rl_model = RLReranker(hidden_dim=128)
    rl_model.load_state_dict(torch.load("method/checkpoints/reranker_best.pt", map_location="cpu", weights_only=True))
    rl_model.eval()

    policy = SubmodularDirectPolicy()

    for th in [None, 0.95, 0.90, 0.85, 0.80]:
        t0 = time.time()
        model_a_scores, model_a_equal = [], []
        model_b_scores, model_b_equal = [], []
        atom_recalls = []

        for q in questions:
            qid = q["qid"]
            if qid not in ak:
                continue
            qtype = q.get("qtype", "")
            cand_ids, q_emb, cand_embs, retrieval_feats = retrieve_candidates(
                retriever, all_sigs, q["question"], qtype, TOP_K_CANDIDATES, threshold=th
            )
            if not cand_ids:
                continue

            entry = ak[qid]
            atoms = compute_credits(entry, run_obj)
            covered = sum(1 for atom in atoms if any(cand in atom for cand in cand_ids))
            atom_recalls.append(covered / max(len(atoms), 1))

            struct_feats = np.stack([feats.get(bid, np.zeros(15, dtype=np.float32)) for bid in cand_ids])
            a_ranked = rl_model.select_mmr(cand_ids, q_emb, cand_embs, struct_feats, retrieval_feats, tokens_map)
            r_a = score_query(entry, a_ranked, run_obj)
            r_a_eq = score_query_equal(entry, a_ranked, run_obj)
            model_a_scores.append(r_a["score"])
            model_a_equal.append(r_a_eq["score"])

            b_ranked = policy.rank(
                cand_ids, retrieval_feats, struct_feats, tokens_map,
                question=q["question"], qtype=qtype, run_text=run_obj["text"]
            )
            r_b = score_query(entry, b_ranked, run_obj)
            r_b_eq = score_query_equal(entry, b_ranked, run_obj)
            model_b_scores.append(r_b["score"])
            model_b_equal.append(r_b_eq["score"])

        label = f"Threshold {th}" if th is not None else "Exact SHA (Baseline)"
        print(f"{label:22s} | Atom Recall: {np.mean(atom_recalls):.4f} | "
              f"Model A (RLReranker): {np.mean(model_a_scores):.4f}/{np.mean(model_a_equal):.4f} | "
              f"Model B (Direct): {np.mean(model_b_scores):.4f}/{np.mean(model_b_equal):.4f} | {time.time()-t0:.1f}s")

