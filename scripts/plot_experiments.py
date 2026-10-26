import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Generate comparison chart for Query Attribute Conditioning & Shallow Tree Ranker experiments.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("experiments_summary.json") as f:
    data = json.load(f)

folds = data["folds"]
domains = [f["domain"].replace("_", "\n") for f in folds]

baseline = [f["baseline"] for f in folds]
w_global = [f["w_global"] for f in folds]
s1a_bilinear = [f["s1a_bilinear_final"] for f in folds]
s1b_cluster = [f["s1b_cluster_final"] for f in folds]
s2a_tree = [f["s2a_pure_lgbm"] for f in folds]
s2b_hybrid = [f["s2b_hybrid_lgbm"] for f in folds]

x = np.arange(len(domains))
width = 0.14

fig, ax = plt.subplots(figsize=(15, 6))

ax.bar(x - 2.5*width, baseline, width, label=f"First-Stage Base ({data['mean_baseline']:.4f})", color="#9E9E9E", alpha=0.85)
ax.bar(x - 1.5*width, w_global, width, label=f"Global Direct Policy ({data['mean_w_global']:.4f})", color="#2196F3", alpha=0.85)
ax.bar(x - 0.5*width, s1a_bilinear, width, label=f"S1A: Bilinear Residual ({data['mean_s1a_bilinear_final']:.4f})", color="#9C27B0", alpha=0.85)
ax.bar(x + 0.5*width, s1b_cluster, width, label=f"S1B: Cluster Residual ({data['mean_s1b_cluster_final']:.4f})", color="#4CAF50", alpha=0.85)
ax.bar(x + 1.5*width, s2a_tree, width, label=f"S2A: Pure LightGBM ({data['mean_s2a_pure_lgbm']:.4f})", color="#FF9800", alpha=0.85)
ax.bar(x + 2.5*width, s2b_hybrid, width, label=f"S2B: Hybrid Tree ({data['mean_s2b_hybrid_lgbm']:.4f})", color="#E91E63", alpha=0.85)

ax.axhline(0.5700, color="red", linestyle="--", linewidth=1.5, label="Benchmark Target (0.5700)")
ax.axhline(data['mean_s1b_cluster_final'], color="#4CAF50", linestyle=":", linewidth=1.2, label=f"Peak 10-Fold CV ({data['mean_s1b_cluster_final']:.4f})")

ax.set_ylabel("Benchmark Score (Nugget Recall across Budgets)", fontsize=11, fontweight="bold")
ax.set_title("10-Fold Cross-Validation: Query Attribute Conditioning & Shallow Tree L2R vs Baselines", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=9)
ax.set_ylim(0.28, 0.62)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("query_tree_experiments_comparison.png", dpi=200)
print("Saved query_tree_experiments_comparison.png successfully!")
