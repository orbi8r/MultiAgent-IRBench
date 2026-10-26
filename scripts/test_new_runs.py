import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
test_new_runs.py — Test the calibrated Master Submodular Direct Policy on the 5 new rebuilt runs.
Strictly test-only (no training).
Computes both First-Stage Baseline and Master Model performance using the official BAR scorer (score.py).
"""
import json
import time
from pathlib import Path
import numpy as np

from method.pipeline import prepare_run, infer_run, save_submission
from method.reranker import SubmodularDirectPolicy
from score import score_run, load_run

NEW_RUNS = [
    "due_diligence",
    "executive_assistants",
    "help_desk",
    "on_call",
    "textbook_team",
]

RESULTS_DIR = Path("method/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("TESTING MASTER RERANKER ON 5 NEW REBUILT RUNS (ZERO TRAINING)")
    print("=" * 70)

    policy = SubmodularDirectPolicy()
    print("Loaded policy with master weights:", np.round(policy.weights, 3))

    results = {
        "runs": {},
        "summary": {}
    }

    fs_main_scores = []
    fs_equal_scores = []
    model_main_scores = []
    model_equal_scores = []

    for r in NEW_RUNS:
        print(f"\n{'='*60}")
        print(f"Evaluating Domain: {r}")
        print(f"{'='*60}")

        t0 = time.time()
        run_data = prepare_run(r)
        print(f"  Prepared {r} in {time.time()-t0:.2f}s")

        run_obj = load_run(f"runs/{r}")
        key = json.load(open(f"runs/{r}/answer_key.json"))
        retriever = run_data["retriever"]
        questions = run_data["questions"]

        # 1. First-Stage Baseline (Order directly from first-stage retrieval)
        print("  Running First-Stage Baseline retrieval...")
        sub_fs = {
            q["qid"]: retriever.retrieve(q["question"], top_k=200, qtype=q.get("qtype", ""))
            for q in questions
        }
        rep_fs = score_run(f"runs/{r}", sub_fs, key=key, run=run_obj)
        fs_main = rep_fs["overall"]["score"]
        fs_equal = rep_fs["equal_credit"]["overall"]["score"]
        print(f"  --> First-Stage Baseline: Main={fs_main:.4f} | Equal Credit={fs_equal:.4f}")

        # 2. Model Inference (Master Direct Policy with Location-Path Boost + Micro-Budget Reordering)
        print("  Running Master Model inference...")
        t_inf = time.time()
        sub_model = infer_run(run_data, rl_model=policy)
        print(f"  --> Inferred {len(sub_model)} queries in {time.time()-t_inf:.2f}s")

        # Save submission
        sub_path = RESULTS_DIR / f"submission_{r}.json"
        save_submission(sub_model, sub_path)

        # Score with official score_run
        rep_model = score_run(f"runs/{r}", sub_model, key=key, run=run_obj)
        model_main = rep_model["overall"]["score"]
        model_equal = rep_model["equal_credit"]["overall"]["score"]
        print(f"  --> Master Model:        Main={model_main:.4f} | Equal Credit={model_equal:.4f}")
        print(f"  --> Net Gain:            Main={model_main - fs_main:+.4f} | Equal Credit={model_equal - fs_equal:+.4f}")

        # Save official report
        rep_path = RESULTS_DIR / f"report_{r}.json"
        with open(rep_path, "w") as f:
            json.dump(rep_model, f, indent=2)

        # Record metrics
        fs_main_scores.append(fs_main)
        fs_equal_scores.append(fs_equal)
        model_main_scores.append(model_main)
        model_equal_scores.append(model_equal)

        results["runs"][r] = {
            "n_questions": len(questions),
            "baseline": {
                "main_score": fs_main,
                "equal_credit": fs_equal,
            },
            "model": {
                "main_score": model_main,
                "equal_credit": model_equal,
                "answered_within_125": rep_model["overall"].get("answered_within_125", 0.0),
                "answered_within_2k": rep_model["overall"].get("answered_within_2000", 0.0),
                "answered_within_32k": rep_model["overall"].get("answered_within_32000", 0.0),
                "by_qtype": rep_model.get("by_qtype", {}),
            },
            "lift": {
                "main_lift": model_main - fs_main,
                "equal_credit_lift": model_equal - fs_equal,
            }
        }

    # Summary
    results["summary"] = {
        "mean_baseline_main": float(np.mean(fs_main_scores)),
        "mean_baseline_equal": float(np.mean(fs_equal_scores)),
        "mean_model_main": float(np.mean(model_main_scores)),
        "mean_model_equal": float(np.mean(model_equal_scores)),
        "mean_lift_main": float(np.mean(model_main_scores) - np.mean(fs_main_scores)),
        "mean_lift_equal": float(np.mean(model_equal_scores) - np.mean(fs_equal_scores)),
    }

    with open(RESULTS_DIR / "five_new_runs_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL 5 NEW RUNS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Domain':24s} | {'Base Main':10s} | {'Model Main':10s} | {'Gain':8s} | {'Base Eq':10s} | {'Model Eq':10s}")
    print("-" * 75)
    for r in NEW_RUNS:
        m = results["runs"][r]
        print(f"{r:24s} | {m['baseline']['main_score']:.4f}     | {m['model']['main_score']:.4f}     | {m['lift']['main_lift']:+.4f}  | {m['baseline']['equal_credit']:.4f}    | {m['model']['equal_credit']:.4f}")
    print("-" * 75)
    s = results["summary"]
    print(f"{'OVERALL MEAN':24s} | {s['mean_baseline_main']:.4f}     | {s['mean_model_main']:.4f}     | {s['mean_lift_main']:+.4f}  | {s['mean_baseline_equal']:.4f}    | {s['mean_model_equal']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
