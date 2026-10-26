"""
Comprehensive Threshold Sweep on Training Runs:
1. library_maintainers (47 queries)
2. literature_review (50 queries)
3. online_shop (49 queries)

Compares Exact SHA vs MinHash at thresholds [0.95, 0.90, 0.85, 0.80]
for BOTH Model A (RLReranker) and Model B (SubmodularDirectPolicy).
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from method.config import RUNS_DIR, TOP_K_CANDIDATES
from method.graph_builder import load_run, build_graph
from method.features import extract_all_block_features, normalise_features
from method.first_stage import FirstStageRetriever
from method.minhash import MinHashEngine
from method.reranker import RLReranker, SubmodularDirectPolicy
from score import score_query, score_query_equal, credits as compute_credits, load_run as score_load_run

TRAIN_RUNS = ["library_maintainers", "literature_review", "online_shop"]

def run_experiment():
    rl_model = RLReranker(hidden_dim=128)
    rl_model.load_state_dict(torch.load("method/checkpoints/reranker_best.pt", map_location="cpu", weights_only=True))
    rl_model.eval()

    policy = SubmodularDirectPolicy()

    # Pre-load data and retrievers
    prepared = {}
    for run_name in TRAIN_RUNS:
        print(f"Loading {run_name}...")
        run_path = RUNS_DIR / run_name
        events, blocks, questions = load_run(run_path)
        run_obj = score_load_run(str(run_path))
        with open(run_path / "answer_key.json") as f:
            ak = json.load(f)
        tokens_map = {e["id"]: e["tokens"] for e in events}
        tokens_map.update({b["id"]: b["tokens"] for b in blocks})
        graph = build_graph(events, blocks)
        feats = normalise_features(extract_all_block_features(blocks, events, graph))
        retriever = FirstStageRetriever(blocks, feats, events=events, use_dense=True)

        engine = MinHashEngine(num_perm=128, threshold=0.85)
        sigs = engine.compute_all(blocks)

        prepared[run_name] = {
            "events": events, "blocks": blocks, "questions": questions,
            "run_obj": run_obj, "ak": ak, "tokens_map": tokens_map,
            "feats": feats, "retriever": retriever, "sigs": sigs,
            "engine": engine
        }

    thresholds = [None, 0.95, 0.90, 0.85, 0.80]

    for th in thresholds:
        th_label = f"MinHash θ={th}" if th is not None else "Exact SHA (Baseline)"
        print(f"\n{'='*70}\nEVALUATING: {th_label}\n{'='*70}")

        macro_a_main, macro_a_eq = [], []
        macro_b_main, macro_b_eq = [], []
        macro_recall = []

        for run_name in TRAIN_RUNS:
            data = prepared[run_name]
            retriever = data["retriever"]
            sigs = data["sigs"]
            engine = data["engine"]
            ak = data["ak"]
            run_obj = data["run_obj"]
            tokens_map = data["tokens_map"]
            feats = data["feats"]

            # Temporarily inject minhash logic
            # retrieve_with_scores with custom threshold
            a_scores, a_eq = [], []
            b_scores, b_eq = [], []
            recalls = []

            for q in data["questions"]:
                qid = q["qid"]
                if qid not in ak:
                    continue
                qtype = q.get("qtype", "")

                # Retrieve candidates with exact SHA or MinHash
                comb, b_scores_dict, f_scores_dict, sims = retriever._get_comb_scores(q["question"], qtype) if hasattr(retriever, "_get_comb_scores") else (None, None, None, None)
                # Since _get_comb_scores isn't an attribute, we use retrieve_with_scores
                # but let's test directly with helper
                pass

if __name__ == "__main__":
    pass
