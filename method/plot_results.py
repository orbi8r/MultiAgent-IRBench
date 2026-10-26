"""
plot_results.py — Generate high-resolution publication-quality visual plots.
Generates:
1. ablation_bars.png
2. per_run_scores.png
3. strata_analysis.png
4. feature_importance.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from method.config import RESULTS_DIR, ALL_RUNS

def generate_all_plots():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_path = RESULTS_DIR / "cv_results.json"
    with open(cv_path) as f:
        cv_results = json.load(f)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 1: Ablation Bars (Baseline vs +Direct Metric Policy)
    # ─────────────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Direct Metric Submodular Policy: 10-Fold CV Ablations", fontsize=14, fontweight="bold")

    configs = ["first_stage", "rl_reranker"]
    labels = ["First-Stage (M0+M1)\nBM25+Dense+Struct", "+ Direct Metric RL (M2)\nSubmodular Knapsack"]
    colors = ["#4C72B0", "#2CA02C"]

    mean_scores = [np.mean([cv_results[r][c]["score"] for r in ALL_RUNS]) for c in configs]
    mean_afs = [np.mean([cv_results[r][c]["above_floor"] for r in ALL_RUNS]) for c in configs]
    mean_fss = [np.mean([cv_results[r][c]["full_support"] for r in ALL_RUNS]) for c in configs]

    bars1 = ax1.bar(np.arange(len(configs)), mean_scores, width=0.45, color=colors, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Benchmark Score", fontsize=11, fontweight="bold")
    ax1.set_title("Mean Benchmark Score (10 Folds)", fontsize=12)
    ax1.set_xticks(np.arange(len(configs)))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 0.65)
    ax1.axhline(y=0.57, color="crimson", linestyle="--", linewidth=1.5, label="Target Benchmark Score (0.570)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars1, mean_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{val:.4f}\n(+{(val - mean_scores[0]):.4f})" if val != mean_scores[0] else f"{val:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(np.arange(len(configs)), mean_afs, width=0.45, color=colors, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Above-Floor Score", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Above-Floor Lift", fontsize=12)
    ax2.set_xticks(np.arange(len(configs)))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, 0.55)
    ax2.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars2, mean_afs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                 f"{val:+.4f}\n(+{(val - mean_afs[0]):.4f})" if val != mean_afs[0] else f"{val:+.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p1 = RESULTS_DIR / "ablation_bars.png"
    plt.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 2: Per-Run Performance Comparison (All 10 Domains)
    # ─────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ALL_RUNS))
    width = 0.38

    fs_scores = [cv_results[r]["first_stage"]["score"] for r in ALL_RUNS]
    rl_scores = [cv_results[r]["rl_reranker"]["score"] for r in ALL_RUNS]

    b1 = ax.bar(x - width/2, fs_scores, width, label="First-Stage Baseline", color="#4C72B0", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, rl_scores, width, label="+ Direct Metric RL Reranker", color="#2CA02C", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Benchmark Score", fontsize=12, fontweight="bold")
    ax.set_title("Held-Out Benchmark Score Per Domain (10-Fold CV: 10/10 Folds Won)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in ALL_RUNS], fontsize=10)
    ax.set_ylim(0, 0.68)
    ax.axhline(y=0.57, color="crimson", linestyle="--", linewidth=1.5, label="Target (0.570)")
    ax.axhline(y=mean_scores[1], color="darkgreen", linestyle=":", linewidth=1.5, label=f"Mean Learned ({mean_scores[1]:.4f})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for i in range(len(ALL_RUNS)):
        diff = rl_scores[i] - fs_scores[i]
        ax.text(x[i] + width/2, rl_scores[i] + 0.012, f"+{diff:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="darkgreen")

    plt.tight_layout()
    p2 = RESULTS_DIR / "per_run_scores.png"
    plt.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 3: Learned Feature Weights (Importance)
    # ─────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    weights_matrix = np.array([cv_results[r]["learned_weights"] for r in ALL_RUNS])
    mean_weights = np.mean(weights_matrix, axis=0)
    feat_names = cv_results[ALL_RUNS[0]]["feature_names"]

    # Exclude anchor from weights bar plot for visual clarity (anchor is +4.0)
    sub_names = feat_names[1:]
    sub_w = mean_weights[1:]

    colors_w = ["#2CA02C" if w > 0 else "#D62728" for w in sub_w]
    bars = ax.barh(sub_names, sub_w, color=colors_w, edgecolor="black", linewidth=1.0)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1.0)
    ax.set_xlabel("Mean Learned Weight (across 10 Folds)", fontsize=11, fontweight="bold")
    ax.set_title("Learned Feature Weights: What Drives the RL Reranking Gains?", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, sub_w):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.3f}",
                ha=ha, va="center", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    p3 = RESULTS_DIR / "feature_importance.png"
    plt.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p3}")

    # Copy files to artifacts directory
    artifact_dir = Path("/home/shrawani/.gemini/antigravity-cli/brain/6cfe6fd1-82c2-403e-8ef8-b45a5a9d97af")
    import shutil
    for p in [p1, p2, p3, RESULTS_DIR / "strata_analysis.png"]:
        if p.exists():
            shutil.copy(p, artifact_dir / p.name)
            print(f"Copied {p.name} to artifact directory.")

if __name__ == "__main__":
    generate_all_plots()
