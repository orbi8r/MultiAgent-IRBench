import json
import numpy as np
import sys
from pathlib import Path
sys.path.append(".")
from method.pipeline import prepare_run, build_training_items
from score import BUDGETS, MIN_COST

# F. Per-domain analysis is already done. Let's do the rest for multiple domains to be sure, maybe library_maintainers and debate_panel.
runs_to_test = ["library_maintainers", "debate_panel"]

try:
    for run_name in runs_to_test:
        print(f"\n=====================================")
        print(f"Analyzing {run_name}...")
        from method.graph_builder import load_run, load_answer_key
        
        answer_key = load_answer_key(Path("runs") / run_name)
        run_data = prepare_run(run_name)
        train_items = build_training_items(run_data, answer_key, top_k=200)
        
        # A, D, E Analysis
        print(f"--- A, D, E Analysis for {run_name} ---")
        
        n_qs = len(train_items)
        nuggets_in_top10 = 0
        nuggets_in_top200 = 0
        total_nuggets = 0
        
        perfect_score_sum = 0
        fs_score_sum = 0
        
        # Budget-level breakdown
        budget_contributions = {B: 0.0 for B in BUDGETS}
        budget_counts = {B: 0 for B in BUDGETS}
        
        for item in train_items:
            qid = item["qid"]
            cand_ids = item["block_ids"]
            ans = item["answer_entry"]
            tokens_map = item["tokens_map"]
            nuggets = ans["nuggets"]
            
            # D. Answer distribution
            total_nuggets += len(nuggets)
            for nid, n_dict in nuggets.items():
                found_10 = any(eid in cand_ids[:10] for eid in n_dict if n_dict[eid] > 0)
                found_200 = any(eid in cand_ids for eid in n_dict if n_dict[eid] > 0)
                if found_10: nuggets_in_top10 += 1
                if found_200: nuggets_in_top200 += 1
                
            # A. Oracle score
            val_map = {}
            for eid in cand_ids:
                val = max([vals.get(eid, 0.0) for vals in nuggets.values()] + [0.0])
                val_map[eid] = val
                
            perfect_ranked = sorted(cand_ids, key=lambda x: -val_map[x])
            from score import score_query
            perfect_res = score_query(ans, perfect_ranked, tokens_map)
            perfect_score_sum += perfect_res["score"]
            
            fs_res = score_query(ans, cand_ids, tokens_map)
            fs_score_sum += fs_res["score"]
            
            # E. Budget-level analysis
            # Let's see how much score comes from rank 1 for each budget
            from score import cut
            for idx, B in enumerate(BUDGETS):
                reach = sum(max((v for e, v in vals.items() if max(tokens_map.get(e, MIN_COST), MIN_COST) <= B), default=0.0)
                            for vals in ans["nuggets"].values())
                if reach > 0:
                    kept, used = cut(cand_ids, tokens_map, B)
                    if kept:
                        # Full score for this budget
                        best = [max((vals.get(e, 0.0) for e in kept), default=0.0) for vals in ans["nuggets"].values()]
                        b_score = sum(best) / reach
                        
                        # Score from rank 1 only
                        rank1_kept = kept[:1]
                        rank1_best = [max((vals.get(e, 0.0) for e in rank1_kept), default=0.0) for vals in ans["nuggets"].values()]
                        rank1_score = sum(rank1_best) / reach
                        
                        budget_contributions[B] += (rank1_score / max(1e-9, b_score))
                        budget_counts[B] += 1
            
        print(f"Total nuggets: {total_nuggets}")
        print(f"Nuggets in Top 10: {nuggets_in_top10} ({nuggets_in_top10/max(1,total_nuggets):.2%})")
        print(f"Nuggets in Top 200: {nuggets_in_top200} ({nuggets_in_top200/max(1,total_nuggets):.2%})")
        print(f"First Stage Score: {fs_score_sum/n_qs:.4f}")
        print(f"Oracle Score (Top 200 sorted by val): {perfect_score_sum/n_qs:.4f}")
        print("Fraction of score from Rank 1 item by budget:")
        for B in BUDGETS:
            if budget_counts[B] > 0:
                print(f"  Budget {B}: {budget_contributions[B]/budget_counts[B]:.2%}")
        
        # B. Feature analysis
        print(f"--- B. Feature Analysis for {run_name} ---")
        retrieval_feats_all = []
        for item in train_items:
            retrieval_feats_all.append(item["retrieval_feats"])
        if retrieval_feats_all:
            rf = np.concatenate(retrieval_feats_all, axis=0)
            print("Retrieval Features Means:", rf.mean(axis=0))
            print("Retrieval Features Std:", rf.std(axis=0))
            print("Correlation with rank (Feature 4 is inverse rank):")
            for i in range(5):
                corr = np.corrcoef(rf[:, i], rf[:, 4])[0, 1]
                print(f"  Feature {i} corr with inverse rank: {corr:.4f}")
            
        # C. Gradient Analysis
        print(f"--- C. Gradient Analysis for {run_name} ---")
        from method.rl_trainer import RLTrainer
        
        avg_marginal_rewards = []
        for item in train_items:
            cand_ids = item["block_ids"]
            ans = item["answer_entry"]
            t_map = item["tokens_map"]
            k_trunc = min(50, len(cand_ids))
            if k_trunc == 0: continue
            marginals = RLTrainer.compute_marginal_rewards(cand_ids[:k_trunc], ans, t_map, k_trunc)
            # pad to 50
            if len(marginals) < 50:
                marginals = np.pad(marginals, (0, 50 - len(marginals)))
            avg_marginal_rewards.append(marginals)
            
        if avg_marginal_rewards:
            avg_m = np.mean(avg_marginal_rewards, axis=0)
            print("Average Marginal Rewards at Top 10 Positions:")
            print(np.round(avg_m[:10], 5))
            
except Exception as e:
    import traceback
    traceback.print_exc()

