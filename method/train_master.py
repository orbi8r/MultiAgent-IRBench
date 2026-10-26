"""
train_master.py — Train the definitive Master Submodular Direct Policy on 100% of all benchmark data.
"""
import argparse
import json
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from method.config import ALL_RUNS, CHECKPOINTS_DIR, RESULTS_DIR, TOP_K_CANDIDATES
from method.reranker import SubmodularDirectPolicy
from method.train import get_or_build_run_items
import numpy as np
from score import score_query, score_query_equal, load_run


def parse_args():
    parser = argparse.ArgumentParser(description="Master Model Training on 100% of Benchmark Data")
    parser.add_argument("--top_k", type=int, default=TOP_K_CANDIDATES)
    parser.add_argument("--n_iter", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("MASTER DIRECT POLICY TRAINING (100% OF BENCHMARK DATA)")
    print("=" * 60)

    # 1. Accumulate all cached items
    all_items = []
    per_run_items = {}
    from score import credits as compute_credits
    t0 = time.time()
    for run_name in ALL_RUNS:
        items = get_or_build_run_items(run_name, top_k=args.top_k)
        run_obj = load_run(f"runs/{run_name}")
        for it in items:
            it["credits"] = compute_credits(it["answer_entry"], run_obj)
            it["run_text"] = run_obj["text"]
        per_run_items[run_name] = items
        all_items.extend(items)
        print(f"  [{run_name}] Accumulated {len(items)} items. Total: {len(all_items)}")

    print(f"\nPooled {len(all_items)} total questions across {len(ALL_RUNS)} domains in {time.time()-t0:.1f}s.")

    # 2. Evaluate Baseline
    policy = SubmodularDirectPolicy()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    master_ckpt_path = CHECKPOINTS_DIR / "direct_policy_master.json"

    # 3. Fit policy on all 498 questions
    print("\nFitting Master Direct Policy (Coordinate Ascent)...")
    t_fit = time.time()
    train_score = policy.fit(all_items, n_iter=args.n_iter)
    policy.save(master_ckpt_path)
    print(f"Master policy fit in {time.time()-t_fit:.1f}s. Final score: {train_score:.4f}")
    print(f"Saved master checkpoint to {master_ckpt_path.name}")

    # 4. Final Evaluation per run and overall
    print("\n" + "=" * 60)
    print("MASTER MODEL BENCHMARK PERFORMANCE ACROSS ALL ACTIVE DOMAINS")
    print("=" * 60)
    per_run_eval = {}
    for run_name in ALL_RUNS:
        items = per_run_items[run_name]
        run_obj = load_run(f"runs/{run_name}")
        scores, equal_scores = [], []
        for it in items:
            ranked = policy.rank(
                it["block_ids"], it["retrieval_feats"], it["struct_feats"], it["tokens_map"],
                question=it.get("question", ""), qtype=it.get("qtype", ""), run_text=run_obj["text"]
            )
            r = score_query(it["answer_entry"], ranked, run_obj)
            req = score_query_equal(it["answer_entry"], ranked, run_obj)
            scores.append(r["score"])
            equal_scores.append(req["score"])
        m = {
            "score": float(np.mean(scores)),
            "equal_credit": float(np.mean(equal_scores)),
            "n_questions": len(scores)
        }
        per_run_eval[run_name] = m
        print(f"  {run_name:24s} | Main BAR: {m['score']:.4f} | Equal Credit: {m['equal_credit']:.4f}")

    overall = {
        "score": float(np.mean([m["score"] for m in per_run_eval.values()])),
        "equal_credit": float(np.mean([m["equal_credit"] for m in per_run_eval.values()])),
        "learned_weights": [float(w) for w in policy.weights],
        "feature_names": list(policy.FEATURE_NAMES),
    }
    print("=" * 60)
    print(f"MASTER BENCHMARK MAIN SCORE : {overall['score']:.4f}")
    print(f"MASTER EQUAL CREDIT SCORE   : {overall['equal_credit']:.4f}")
    print("=" * 60)

    # 5. Save master evaluation
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "master_model_eval.json", "w") as f:
        json.dump({"overall": overall, "per_run": per_run_eval}, f, indent=2)

    return overall


if __name__ == "__main__":
    main()
