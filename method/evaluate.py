"""
evaluate.py — Evaluation, ablations, strata analysis, and learning curve plots.

Usage:
    python3 -m method.evaluate --results method/results/cv_results.json
    python3 -m method.evaluate --run on_call --compare  (compare all ablations)
    python3 -m method.evaluate --plot_curves            (learning curve plots)
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from method.config import ALL_RUNS, RESULTS_DIR, RUNS_DIR


# ─── Score a submission from file ─────────────────────────────────────────

def score_from_file(run_dir: Path, sub_path: Path) -> dict:
    """Score a submission file against a run's answer key."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from score import score_query, load_tokens
    import json as _json

    key = _json.load(open(run_dir / "answer_key.json"))
    sub = _json.load(open(sub_path))
    tokens = load_tokens(run_dir / "events.jsonl", run_dir / "blocks.jsonl")

    per_q = {q: score_query(entry, sub.get(q, []), tokens) for q, entry in key.items()}
    scores = [v["score"] for v in per_q.values()]
    above_floors = [v["above_floor"] for v in per_q.values()]
    full_supports = [v["full_support"] for v in per_q.values()]

    return {
        "score": sum(scores) / len(scores),
        "above_floor": sum(above_floors) / len(above_floors),
        "full_support": sum(full_supports) / len(full_supports),
        "per_query": per_q,
    }


# ─── Strata analysis ──────────────────────────────────────────────────────

def strata_analysis(cv_results: dict, histories: dict = None) -> dict:
    """
    Per-stratum analysis across folds.
    Stratifies by: qtype, spread (single/multi event), agents (single/multi agent)
    """
    strata_scores = defaultdict(list)

    for fold, res in cv_results.items():
        run_dir = RUNS_DIR / fold
        if not (run_dir / "answer_key.json").exists():
            continue
        key = json.load(open(run_dir / "answer_key.json"))
        sub_path = RESULTS_DIR / f"submission_{fold}.json"
        if not sub_path.exists():
            continue

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from score import score_query, load_tokens
        tokens = load_tokens(run_dir / "events.jsonl", run_dir / "blocks.jsonl")
        sub = json.load(open(sub_path))

        for qid, entry in key.items():
            strata = entry.get("strata", {})
            qtype = entry.get("qtype", "unknown")
            ranked = sub.get(qid, [])
            result = score_query(entry, ranked, tokens)

            strata_scores[f"qtype={qtype}"].append(result["score"])
            for k, v in strata.items():
                if v:
                    strata_scores[f"{k}={v}"].append(result["score"])

    return {
        stratum: {
            "mean_score": sum(scores) / len(scores),
            "n": len(scores),
        }
        for stratum, scores in strata_scores.items()
    }


# ─── Ablation table ───────────────────────────────────────────────────────

def print_ablation_table(cv_results: dict):
    """Print a formatted ablation table across all folds."""
    configs = ["first_stage", "rl_reranker", "rl_gnn", "full_system"]
    config_labels = {
        "first_stage": "BM25+Dense+Struct (M0+M1)",
        "rl_reranker": "+ RL Reranker (M2)",
        "rl_gnn":      "+ GNN Reranker (M3)",
        "full_system": "+ Budget-Aware (M4)",
    }

    print("\n" + "="*80)
    print("ABLATION RESULTS (Leave-One-Run-Out Cross-Validation)")
    print("="*80)
    print(f"{'Configuration':<30} {'Score':>8} {'AboveFloor':>12} {'FullSupport':>13} {'N Folds':>8}")
    print("-"*80)

    for config in configs:
        scores, af, fs = [], [], []
        for fold, res in cv_results.items():
            if config in res:
                scores.append(res[config]["score"])
                af.append(res[config]["above_floor"])
                fs.append(res[config]["full_support"])
        if scores:
            label = config_labels.get(config, config)
            print(f"{label:<30} {sum(scores)/len(scores):>8.4f} "
                  f"{sum(af)/len(af):>+12.4f} "
                  f"{sum(fs)/len(fs):>13.4f} "
                  f"{len(scores):>8d}")

    print("="*80)


# ─── Learning curve plots ──────────────────────────────────────────────────

def plot_learning_curves(cv_results: dict, output_dir: Path = RESULTS_DIR):
    """Generate and save learning curve plots using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load histories from histories.json (written by train.py) if available
    histories_path = output_dir / "histories.json"
    external_histories = {}
    if histories_path.exists():
        external_histories = json.load(open(histories_path))

    # Collect RL training histories across folds
    all_histories = []
    fold_names = []
    for fold, res in cv_results.items():
        hist = res.get("rl_history") or external_histories.get(fold, {}).get("rl")
        if hist and "epoch" in hist and len(hist["epoch"]) > 0:
            all_histories.append(hist)
            fold_names.append(fold)

    if not all_histories:
        print("No training history found — skipping learning curves")
        return

    # ─── Figure 1: Main learning curves ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RL Reranker Training Curves (Leave-One-Run-Out CV)", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    # Plot 1: Training loss
    ax = axes[0, 0]
    for hist, name, color in zip(all_histories, fold_names, colors):
        epochs = hist["epoch"]
        loss = hist["loss"]
        ax.plot(epochs, loss, alpha=0.7, color=color, linewidth=1.5, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("REINFORCE Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Training reward
    ax = axes[0, 1]
    for hist, name, color in zip(all_histories, fold_names, colors):
        epochs = hist["epoch"]
        reward = hist["mean_reward"]
        ax.plot(epochs, reward, alpha=0.7, color=color, linewidth=1.5, label=name)
    # Also plot mean across folds
    max_ep = max(len(h["epoch"]) for h in all_histories)
    mean_reward = []
    for ep in range(max_ep):
        vals = [h["mean_reward"][ep] for h in all_histories if ep < len(h["mean_reward"])]
        mean_reward.append(sum(vals) / len(vals))
    ax.plot(range(max_ep), mean_reward, "k--", linewidth=2.5, label="Mean", zorder=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Training Reward")
    ax.set_title("Training Reward (higher = better)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 3: Above-floor during training
    ax = axes[1, 0]
    for hist, name, color in zip(all_histories, fold_names, colors):
        epochs = hist["epoch"]
        af = hist["above_floor"]
        ax.plot(epochs, af, alpha=0.7, color=color, linewidth=1.5, label=name)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Floor")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Above Floor")
    ax.set_title("Training: Score Above Floor")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 4: Temperature annealing + eval scores
    ax = axes[1, 1]
    for hist, name, color in zip(all_histories, fold_names, colors):
        epochs = hist["epoch"]
        temp = hist.get("temperature", [])
        if temp:
            ax.plot(epochs, temp, alpha=0.6, color=color, linewidth=1.5, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Temperature τ")
    ax.set_title("Temperature Annealing (τ)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "learning_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves: {fig_path}")

    # ─── Figure 2: Eval scores over training ──────────────────────────────
    eval_hists = [(h, n, c) for h, n, c in zip(all_histories, fold_names, colors)
                  if h.get("eval_score")]
    if eval_hists:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Evaluation Score Progress During Training", fontsize=13, fontweight="bold")

        ax = axes[0]
        for hist, name, color in eval_hists:
            n_eval = len(hist["eval_score"])
            total_eps = len(hist["epoch"])
            eval_epochs = [e for e in range(total_eps) if (e % 10 == 0 or e == total_eps - 1)][:n_eval]
            if len(eval_epochs) != n_eval:
                eval_epochs = np.linspace(0, max(1, total_eps - 1), n_eval).astype(int).tolist()
            ax.plot(eval_epochs, hist["eval_score"],
                    "o-", alpha=0.8, color=color, linewidth=2, markersize=4, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Eval Score (Held-Out Run)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for hist, name, color in eval_hists:
            if hist.get("eval_above_floor"):
                n_eval = len(hist["eval_above_floor"])
                total_eps = len(hist["epoch"])
                eval_epochs = [e for e in range(total_eps) if (e % 10 == 0 or e == total_eps - 1)][:n_eval]
                if len(eval_epochs) != n_eval:
                    eval_epochs = np.linspace(0, max(1, total_eps - 1), n_eval).astype(int).tolist()
                ax.plot(eval_epochs, hist["eval_above_floor"],
                        "s-", alpha=0.8, color=color, linewidth=2, markersize=4, label=name)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Baseline floor")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Above Floor")
        ax.set_title("Eval Score Above Floor")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path2 = output_dir / "eval_progress.png"
        plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved eval progress: {fig_path2}")

    # ─── Figure 3: Ablation bar chart ─────────────────────────────────────
    plot_ablation_bars(cv_results, output_dir)

    # ─── Figure 4: Per-run performance ────────────────────────────────────
    plot_per_run_scores(cv_results, output_dir)


def plot_ablation_bars(cv_results: dict, output_dir: Path):
    """Bar chart comparing ablation configurations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = ["first_stage", "rl_reranker", "rl_gnn", "full_system"]
    labels = ["M0+M1\n(Struct+Fuse)", "M0-M2\n(+RL)", "M0-M3\n(+GNN)", "Full\n(+Budget)"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    scores = []
    above_floors = []
    for config in configs:
        s = [res[config]["score"] for res in cv_results.values() if config in res]
        af = [res[config]["above_floor"] for res in cv_results.values() if config in res]
        scores.append(sum(s) / len(s) if s else 0)
        above_floors.append(sum(af) / len(af) if af else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ablation Study: Module Contribution", fontsize=13, fontweight="bold")

    x = np.arange(len(configs))
    width = 0.5

    bars1 = ax1.bar(x, scores, width, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Score")
    ax1.set_title("Mean Score Across Folds")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, above_floors, width, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Score Above Floor")
    ax2.set_title("Mean Score Above Query-Blind Floor")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Floor baseline")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars2, above_floors):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{val:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "ablation_bars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation bars: {path}")


def plot_per_run_scores(cv_results: dict, output_dir: Path):
    """Line plot showing scores across different runs/domains."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = ["first_stage", "rl_reranker", "rl_gnn", "full_system"]
    config_colors = {"first_stage": "#4C72B0", "rl_reranker": "#DD8452",
                     "rl_gnn": "#55A868", "full_system": "#C44E52"}
    config_labels = {
        "first_stage": "M0+M1 (Struct+Fuse)",
        "rl_reranker": "M0-M2 (+RL)",
        "rl_gnn":      "M0-M3 (+GNN)",
        "full_system": "Full System",
    }

    fold_names = list(cv_results.keys())
    if not fold_names:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Per-Run Performance Across Domains", fontsize=13, fontweight="bold")

    x = np.arange(len(fold_names))
    width = 0.2

    for i, config in enumerate(configs):
        scores = []
        for fold in fold_names:
            if config in cv_results.get(fold, {}):
                scores.append(cv_results[fold][config]["score"])
            else:
                scores.append(0)
        offset = (i - 1.5) * width
        ax1.bar(x + offset, scores, width, label=config_labels.get(config, config),
                color=config_colors[config], alpha=0.8, edgecolor="white")

    ax1.set_xlabel("Run (Domain)")
    ax1.set_ylabel("Score")
    ax1.set_title("Score by Domain")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace("_", "\n") for f in fold_names], fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Plot above_floor
    for i, config in enumerate(configs):
        afs = []
        for fold in fold_names:
            if config in cv_results.get(fold, {}):
                afs.append(cv_results[fold][config]["above_floor"])
            else:
                afs.append(0)
        offset = (i - 1.5) * width
        ax2.bar(x + offset, afs, width, label=config_labels.get(config, config),
                color=config_colors[config], alpha=0.8, edgecolor="white")

    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Floor")
    ax2.set_xlabel("Run (Domain)")
    ax2.set_ylabel("Above Floor")
    ax2.set_title("Score Above Floor by Domain")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace("_", "\n") for f in fold_names], fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "per_run_scores.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-run scores: {path}")


def plot_strata_analysis(strata_results: dict, output_dir: Path = RESULTS_DIR):
    """Plot per-stratum performance heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not strata_results:
        return

    strata_sorted = sorted(strata_results.items(), key=lambda x: -x[1]["n"])[:20]
    labels = [s for s, _ in strata_sorted]
    scores = [v["mean_score"] for _, v in strata_sorted]
    ns = [v["n"] for _, v in strata_sorted]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.array(scores))
    bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Score")
    ax.set_title("Per-Stratum Performance (Full System)")
    ax.axvline(x=0.3, color="blue", linestyle="--", alpha=0.5, label="Fusion baseline ~0.3")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=8)

    plt.tight_layout()
    path = output_dir / "strata_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved strata analysis: {path}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluation and ablation analysis")
    parser.add_argument("--results", type=str,
                        default=str(RESULTS_DIR / "cv_results.json"),
                        help="Path to CV results JSON")
    parser.add_argument("--plot_curves", action="store_true",
                        help="Generate learning curve plots")
    parser.add_argument("--ablation_table", action="store_true",
                        help="Print ablation table")
    parser.add_argument("--strata", action="store_true",
                        help="Run strata analysis")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run method.train first to generate results.")
        return

    cv_results = json.load(open(results_path))
    print(f"Loaded results for {len(cv_results)} folds")

    print_ablation_table(cv_results)

    if args.plot_curves or True:  # Always generate plots
        plot_learning_curves(cv_results, RESULTS_DIR)

    if args.strata:
        strata_res = strata_analysis(cv_results)
        strata_path = RESULTS_DIR / "strata_results.json"
        json.dump(strata_res, open(strata_path, "w"), indent=2)
        print(f"\nSaved strata results: {strata_path}")
        plot_strata_analysis(strata_res, RESULTS_DIR)

        print("\nPer-Stratum Analysis:")
        for stratum, stats in sorted(strata_res.items(), key=lambda x: -x[1]["n"])[:15]:
            print(f"  {stratum:<35} score={stats['mean_score']:.4f}  n={stats['n']}")


if __name__ == "__main__":
    main()
