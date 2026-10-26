import json
import numpy as np
import sys
from pathlib import Path
sys.path.append(".")
from score import score_query, BUDGETS, MIN_COST, load_tokens
from method.config import ALL_RUNS

total_nuggets = 0
nuggets_in_top10 = 0
nuggets_in_top200 = 0
perfect_score_sum = 0
fs_score_sum = 0
total_qs = 0

budget_contributions = {B: [] for B in BUDGETS}
domain_scores = {}

for run_name in ALL_RUNS:
    sub_path = Path("method/results") / f"submission_{run_name}.json"
    if not sub_path.exists(): continue
    
    with open(sub_path) as f:
        sub = json.load(f)
        
    ans_key_path = Path("runs") / run_name / "answer_key.json"
    with open(ans_key_path) as f:
        key = json.load(f)
        
    events_path = Path("runs") / run_name / "events.jsonl"
    blocks_path = Path("runs") / run_name / "blocks.jsonl"
    tokens = load_tokens(events_path, blocks_path)
    
    run_qs = 0
    run_perfect_score = 0
    run_fs_score = 0
    
    for qid, ans in key.items():
        if qid not in sub: continue
        cand_ids = sub[qid][:200]
        nuggets = ans["nuggets"]
        total_nuggets += len(nuggets)
        
        for nid, n_dict in nuggets.items():
            found_10 = any(eid in cand_ids[:10] for eid in n_dict if n_dict[eid] > 0)
            found_200 = any(eid in cand_ids for eid in n_dict if n_dict[eid] > 0)
            if found_10: nuggets_in_top10 += 1
            if found_200: nuggets_in_top200 += 1
            
        val_map = {}
        for eid in cand_ids:
            val_map[eid] = max([vals.get(eid, 0.0) for vals in nuggets.values()] + [0.0])
            
        perfect_ranked = sorted(cand_ids, key=lambda x: -val_map[x])
        perfect_res = score_query(ans, perfect_ranked, tokens)
        perfect_score_sum += perfect_res["score"]
        run_perfect_score += perfect_res["score"]
        
        # Submissions are usually from full_system. To get first_stage, we need the raw first stage.
        # But wait, did the submission reorder? Let's just evaluate it as is to see what the score is.
        # Actually, if we just want to know Oracle, we can reorder.
        fs_res = score_query(ans, cand_ids, tokens)
        fs_score_sum += fs_res["score"]
        run_fs_score += fs_res["score"]
        run_qs += 1
        total_qs += 1
        
        # Budget-level analysis on the submitted ranking (RL/full_system)
        for B in BUDGETS:
            reach = sum(max((v for e, v in vals.items() if max(tokens.get(e, MIN_COST), MIN_COST) <= B), default=0.0)
                        for vals in ans["nuggets"].values())
            if reach > 0:
                from score import cut
                kept, _ = cut(cand_ids, tokens, B)
                if kept:
                    best = [max((vals.get(e, 0.0) for e in kept), default=0.0) for vals in ans["nuggets"].values()]
                    b_score = sum(best) / reach
                    
                    rank1_kept = kept[:1]
                    rank1_best = [max((vals.get(e, 0.0) for e in rank1_kept), default=0.0) for vals in ans["nuggets"].values()]
                    rank1_score = sum(rank1_best) / reach
                    
                    if b_score > 1e-9:
                        budget_contributions[B].append(rank1_score / b_score)
                        
    domain_scores[run_name] = {
        "oracle": run_perfect_score / run_qs if run_qs else 0,
        "submission": run_fs_score / run_qs if run_qs else 0
    }

print("--- A. Score Ceiling & D. Answer Distribution ---")
print(f"Total Questions: {total_qs}")
print(f"Total Nuggets: {total_nuggets}")
print(f"Nuggets in Top 10: {nuggets_in_top10} ({nuggets_in_top10/total_nuggets:.2%})")
print(f"Nuggets in Top 200: {nuggets_in_top200} ({nuggets_in_top200/total_nuggets:.2%})")
print(f"Average Submission Score (should match RL/full_system): {fs_score_sum/total_qs:.4f}")
print(f"Average Oracle Score (Top 200 sorted by val): {perfect_score_sum/total_qs:.4f}")
print(f"Room for improvement over submission: {(perfect_score_sum - fs_score_sum)/total_qs:.4f}")

print("\n--- E. Budget-level Analysis ---")
print("Fraction of score from Rank 1 item by budget window:")
for B in BUDGETS:
    if budget_contributions[B]:
        print(f"  Budget {B}: {np.mean(budget_contributions[B]):.2%}")

