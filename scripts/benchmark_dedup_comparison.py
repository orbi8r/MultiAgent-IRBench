"""
benchmark_dedup_comparison.py — Full Comparative Evaluation:
Exact SHA-1 Deduplication vs MinHash Near-Duplicate Deduplication.

Evaluates BOTH models:
  - Model A: Deep Neural RL Reranker (hidden_dim=128, reranker_best.pt)
  - Model B: Submodular Direct Policy Search (direct_policy_master.json)

Domains:
  - 5 Held-Out Test Runs (Strictly Unseen / Zero Training):
      due_diligence, executive_assistants, help_desk, on_call, textbook_team
  - 3 Original Training Runs (For Reference):
      library_maintainers, literature_review, online_shop
"""
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from method.config import RUNS_DIR, RESULTS_DIR, TOP_K_CANDIDATES
from method.pipeline import prepare_run, infer_run, save_submission
from method.reranker import RLReranker, SubmodularDirectPolicy
from score import score_run, load_run as score_load_run

TEST_RUNS = [
    "due_diligence",
    "executive_assistants",
    "help_desk",
    "on_call",
    "textbook_team",
]

TRAIN_RUNS = [
    "library_maintainers",
    "literature_review",
    "online_shop",
]

ALL_DOMAINS = TEST_RUNS + TRAIN_RUNS


def run_evaluation(thresholds=[None, 0.85], runs=TEST_RUNS):
    # 1. Load Model A (Deep Neural RL Reranker)
    print("Loading Model A (RLReranker, hidden_dim=128)...")
    model_a = RLReranker(hidden_dim=128)
    model_a_ckpt = torch.load("method/checkpoints/reranker_best.pt", map_location="cpu", weights_only=True)
    model_a.load_state_dict(model_a_ckpt)
    model_a.eval()

    # 2. Load Model B (Submodular Direct Policy)
    print("Loading Model B (SubmodularDirectPolicy)...")
    model_b = SubmodularDirectPolicy()

    results = {}

    for th in thresholds:
        th_key = "exact_sha" if th is None else f"minhash_{int(th*100)}"
        th_title = "Exact SHA-1 (Baseline)" if th is None else f"MinHash Near-Dedup (θ={th})"
        results[th_key] = {"title": th_title, "threshold": th, "runs": {}}

        print(f"\n{'='*75}")
        print(f"STARTING BENCHMARK: {th_title.upper()}")
        print(f"{'='*75}")

        for run_name in runs:
            t0 = time.time()
            run_data = prepare_run(run_name)
            retriever = run_data["retriever"]
            retriever.minhash_threshold = th
            run_obj = score_load_run(str(RUNS_DIR / run_name))

            # Run Model A inference
            sub_a = infer_run(run_data, rl_model=model_a, top_k=TOP_K_CANDIDATES)
            rep_a = score_run(RUNS_DIR / run_name, sub_a, run=run_obj)
            score_a = rep_a["overall"]["score"]
            eq_a = rep_a["equal_credit"]["overall"]["score"]

            # Save Model A submission
            sub_a_path = RESULTS_DIR / f"{run_name}_{th_key}_modelA.json"
            save_submission(sub_a, sub_a_path)

            # Run Model B inference
            sub_b = infer_run(run_data, rl_model=model_b, top_k=TOP_K_CANDIDATES)
            rep_b = score_run(RUNS_DIR / run_name, sub_b, run=run_obj)
            score_b = rep_b["overall"]["score"]
            eq_b = rep_b["equal_credit"]["overall"]["score"]

            # Save Model B submission
            sub_b_path = RESULTS_DIR / f"{run_name}_{th_key}_modelB.json"
            save_submission(sub_b, sub_b_path)

            elapsed = time.time() - t0
            run_res = {
                "model_a_main": score_a,
                "model_a_equal": eq_a,
                "model_b_main": score_b,
                "model_b_equal": eq_b,
                "n_questions": len(sub_a),
                "elapsed_sec": elapsed,
            }
            results[th_key]["runs"][run_name] = run_res

            print(f"  [{run_name:22s}] "
                  f"Model A: Main={score_a:.4f} Equal={eq_a:.4f} | "
                  f"Model B: Main={score_b:.4f} Equal={eq_b:.4f} "
                  f"({elapsed:.1f}s)")

        # Compute averages for this deduplication setting
        a_mains = [results[th_key]["runs"][r]["model_a_main"] for r in runs]
        a_eqs = [results[th_key]["runs"][r]["model_a_equal"] for r in runs]
        b_mains = [results[th_key]["runs"][r]["model_b_main"] for r in runs]
        b_eqs = [results[th_key]["runs"][r]["model_b_equal"] for r in runs]

        results[th_key]["macro_avg"] = {
            "model_a_main": float(np.mean(a_mains)),
            "model_a_equal": float(np.mean(a_eqs)),
            "model_b_main": float(np.mean(b_mains)),
            "model_b_equal": float(np.mean(b_eqs)),
        }

        print(f"\n  --- {th_title} MACRO MEAN ({len(runs)} runs) ---")
        print(f"  Model A (RLReranker):           Main={results[th_key]['macro_avg']['model_a_main']:.4f} | Equal={results[th_key]['macro_avg']['model_a_equal']:.4f}")
        print(f"  Model B (SubmodularDirectPolicy): Main={results[th_key]['macro_avg']['model_b_main']:.4f} | Equal={results[th_key]['macro_avg']['model_b_equal']:.4f}")

    # Save full comparison report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = RESULTS_DIR / "minhash_vs_exact_sha_comparison.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {report_file}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run on all 8 runs (test + train)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.85], help="MinHash thresholds")
    args = parser.parse_args()

    runs_to_eval = ALL_DOMAINS if args.all else TEST_RUNS
    th_list = [None] + args.thresholds
    run_evaluation(thresholds=th_list, runs=runs_to_eval)
