import json
import numpy as np
import sys
from pathlib import Path
sys.path.append(".")

# Patch DenseIndex to make it instantaneous
import method.first_stage
class MockDenseIndex:
    def __init__(self, blocks, *args, **kwargs):
        self.embeddings = np.zeros((len(blocks), 384), dtype=np.float32)
    def encode_query(self, query):
        return np.zeros(384, dtype=np.float32)
    def free(self): pass
method.first_stage.DenseIndex = MockDenseIndex

from method.pipeline import prepare_run, build_training_items
from score import BUDGETS, MIN_COST
from method.graph_builder import load_run, load_answer_key
from method.rl_trainer import RLTrainer

runs_to_test = ["library_maintainers"]

for run_name in runs_to_test:
    print(f"Loading {run_name} for feature/gradient analysis...")
    answer_key = load_answer_key(Path("runs") / run_name)
    run_data = prepare_run(run_name)
    train_items = build_training_items(run_data, answer_key, top_k=200)

    print("\n--- B. Feature Analysis ---")
    retrieval_feats_all = []
    struct_feats_all = []
    for item in train_items:
        retrieval_feats_all.append(item["retrieval_feats"])
        struct_feats_all.append(item["struct_feats"])
        
    if retrieval_feats_all:
        rf = np.concatenate(retrieval_feats_all, axis=0)
        sf = np.concatenate(struct_feats_all, axis=0)
        
        print("Retrieval Features Means:", np.round(rf.mean(axis=0), 4))
        print("Retrieval Features Std: ", np.round(rf.std(axis=0), 4))
        print("Correlation with inverse rank (Feature 4):")
        for i in range(5):
            corr = np.corrcoef(rf[:, i], rf[:, 4])[0, 1]
            print(f"  Feature {i}: {corr:.4f}")
            
    print("\n--- C. Gradient Analysis ---")
    avg_marginal_rewards = []
    for item in train_items:
        cand_ids = item["block_ids"]
        ans = item["answer_entry"]
        t_map = item["tokens_map"]
        k_trunc = min(50, len(cand_ids))
        if k_trunc == 0: continue
        marginals = RLTrainer.compute_marginal_rewards(cand_ids[:k_trunc], ans, t_map, k_trunc)
        
        if len(marginals) < 50:
            marginals = np.pad(marginals, (0, 50 - len(marginals)))
        avg_marginal_rewards.append(marginals)
        
    if avg_marginal_rewards:
        avg_m = np.mean(avg_marginal_rewards, axis=0)
        print("Average Marginal Rewards at Top 50 Positions (first 15 shown):")
        print(np.round(avg_m[:15], 5))
        
        # Calculate sum of marginals to show how concentrated it is
        print(f"Sum of top 10 marginals: {np.sum(avg_m[:10]):.5f}")
        print(f"Sum of next 40 marginals: {np.sum(avg_m[10:50]):.5f}")

