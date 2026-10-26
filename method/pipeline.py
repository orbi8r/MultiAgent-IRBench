"""
pipeline.py — End-to-end pipeline: load run → build graph → retrieve → rerank → submit.

Memory notes:
  - Uses shared MiniLM model (CPU encoding) via first_stage.get_shared_model()
  - Block embeddings stay on CPU numpy; only micro-batches move to GPU during training
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from method.config import (
    ALL_RUNS, CHECKPOINTS_DIR, DEVICE, RESULTS_DIR, RUNS_DIR, TOP_K_CANDIDATES
)
from method.graph_builder import build_graph, load_run, load_answer_key
from method.features import extract_all_block_features, normalise_features
from method.first_stage import FirstStageRetriever
from method.reranker import RLReranker
from method.gnn_reranker import GNNReranker, gnn_rerank
from method.budget_selector import build_submission
from method.oracle import compute_greedy_oracle_sequence


def build_tokens_map(events: list, blocks: list) -> Dict[str, int]:
    """Build {id: tokens} map for all events and blocks."""
    tokens_map = {}
    for e in events:
        tokens_map[e["id"]] = e["tokens"]
    for b in blocks:
        tokens_map[b["id"]] = b["tokens"]
    return tokens_map


def prepare_run(run_name: str) -> dict:
    """
    Load and preprocess a run: graph, features, embeddings.

    Returns a run_data dict with all precomputed artifacts.
    """
    run_dir = RUNS_DIR / run_name
    print(f"\n{'='*60}")
    print(f"Preparing run: {run_name}")
    print(f"{'='*60}")

    t0 = time.time()
    events, blocks, questions = load_run(run_dir)
    print(f"  Loaded {len(events)} events, {len(blocks)} blocks, {len(questions)} questions in {time.time()-t0:.1f}s")

    t1 = time.time()
    graph = build_graph(events, blocks)
    total_edges = sum(len(v) for v in graph["edges"].values())
    print(f"  Built provenance graph: {total_edges} edges in {time.time()-t1:.1f}s")

    t2 = time.time()
    block_features_raw = extract_all_block_features(blocks, events, graph)
    block_features = normalise_features(block_features_raw)
    print(f"  Extracted structural features in {time.time()-t2:.1f}s")

    t3 = time.time()
    retriever = FirstStageRetriever(blocks, block_features, events=events, use_dense=True)
    print(f"  Built first-stage retriever in {time.time()-t3:.1f}s")

    tokens_map = build_tokens_map(events, blocks)
    run_text = {b["id"]: b["text"] for b in blocks}
    run_text.update({e["id"]: e["text"] for e in events})

    return dict(
        run_name=run_name,
        run_dir=run_dir,
        events=events,
        blocks=blocks,
        questions=questions,
        graph=graph,
        block_features=block_features,
        block_features_raw=block_features_raw,
        retriever=retriever,
        tokens_map=tokens_map,
        run_text=run_text,
    )


def build_training_items(
    run_data: dict,
    answer_key: dict,
    top_k: int = TOP_K_CANDIDATES,
) -> list:
    """
    Build training data items for RL trainer from a prepared run.

    Returns list of dicts with all tensors precomputed, including
    first-stage retrieval scoring features (5-d per candidate).
    """
    retriever = run_data["retriever"]
    block_features = run_data["block_features"]
    tokens_map = run_data["tokens_map"]
    questions = run_data["questions"]
    graph = run_data.get("graph", {})

    items = []
    for q in questions:
        qid = q["qid"]
        if qid not in answer_key:
            continue

        # Retrieve candidates with embeddings AND first-stage scoring features
        try:
            cand_ids, q_emb, cand_embs, retrieval_feats = retriever.retrieve_with_scores(
                q["question"], top_k, qtype=q.get("qtype", "")
            )
        except Exception as e:
            print(f"  Warning: retrieval failed for {qid}: {e}")
            continue

        if len(cand_ids) == 0:
            continue

        # Structural features for candidates
        struct_feats = np.stack([
            block_features.get(bid, np.zeros(15, dtype=np.float32))
            for bid in cand_ids
        ])

        try:
            oracle_ids, oracle_gains = compute_greedy_oracle_sequence(
                answer_key[qid], cand_ids, tokens_map, max_items=25
            )
            oracle_indices = [cand_ids.index(b) for b in oracle_ids if b in cand_ids]
        except Exception:
            oracle_indices, oracle_gains = [], []

        items.append(dict(
            qid=qid,
            question=q["question"],
            qtype=q.get("qtype", ""),
            query_emb=q_emb,
            block_embs=cand_embs,
            struct_feats=struct_feats,
            retrieval_feats=retrieval_feats,  # (N, 10) float32
            block_ids=cand_ids,
            tokens_map=tokens_map,
            answer_entry=answer_key[qid],
            graph=graph,
            oracle_indices=oracle_indices,
            oracle_gains=oracle_gains,
        ))

    return items


def infer_run(
    run_data: dict,
    rl_model: Optional[RLReranker] = None,
    gnn_model: Optional[GNNReranker] = None,
    top_k: int = TOP_K_CANDIDATES,
    use_budget_selection: bool = False,
    use_diversity: bool = False,
) -> Dict[str, List[str]]:
    """
    Run full inference pipeline on a prepared run.

    Returns submission dict {qid: [ranked_ids...]}.
    """
    retriever = run_data["retriever"]
    block_features = run_data["block_features"]
    tokens_map = run_data["tokens_map"]
    questions = run_data["questions"]
    graph = run_data["graph"]

    retrieval_results: Dict[str, Tuple[List[str], List[float]]] = {}

    # Pre-build block embeddings lookup once if GNN is used
    all_block_embs = None
    if rl_model is not None and gnn_model is not None and retriever.dense is not None:
        all_block_embs = {
            bid: retriever.dense.embeddings[i]
            for i, bid in enumerate(retriever.block_ids)
        }

    for q in questions:
        qid = q["qid"]
        qtype = q.get("qtype", "")

        if rl_model is not None:
            # Full pipeline with RL reranking — use retrieve_with_scores
            cand_ids, q_emb, cand_embs, retrieval_feats = retriever.retrieve_with_scores(
                q["question"], top_k, qtype=qtype
            )
            if len(cand_ids) == 0:
                retrieval_results[qid] = ([], [])
                continue

            struct_feats = np.stack([
                block_features.get(bid, np.zeros(15, dtype=np.float32))
                for bid in cand_ids
            ])

            if hasattr(rl_model, "rank"):
                reranked_ids = rl_model.rank(
                    cand_ids, retrieval_feats, struct_feats, tokens_map,
                    question=q["question"], qtype=qtype, run_text=run_data.get("run_text")
                )
                final_scores = rl_model.score_candidates(
                    cand_ids, retrieval_feats, struct_feats, tokens_map,
                    question=q["question"], qtype=qtype, run_text=run_data.get("run_text")
                )
            else:
                rl_scores = rl_model.score_candidates(q_emb, cand_embs, struct_feats, retrieval_feats)

                if gnn_model is not None:
                    # GNN reranking
                    reranked_ids, final_scores = gnn_rerank(
                        cand_ids, rl_scores, cand_embs, struct_feats,
                        graph, gnn_model, all_block_embs, block_features,
                        query_emb=q_emb,
                    )
                else:
                    # Contextual Sequential MMR selection
                    reranked_ids = rl_model.select_mmr(
                        cand_ids, q_emb, cand_embs, struct_feats, retrieval_feats, tokens_map
                    )
                    final_scores = rl_scores

        else:
            # First-stage only
            cand_ids = retriever.retrieve(q["question"], top_k, qtype=qtype)
            final_scores = np.ones(len(cand_ids), dtype=np.float32) * (
                len(cand_ids) - np.arange(len(cand_ids))
            ) / len(cand_ids)
            reranked_ids = cand_ids

        retrieval_results[qid] = (reranked_ids, list(final_scores))

    if use_budget_selection:
        submission = build_submission(
            questions, retrieval_results, tokens_map,
            use_diversity=use_diversity
        )
    else:
        submission = {qid: ids for qid, (ids, _) in retrieval_results.items()}

    return submission


def save_submission(submission: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f)
    print(f"Saved submission to {path}")
