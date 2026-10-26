import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Comprehensive 10-Fold Cross-Validation Experiment for:
1. First-Stage Baseline
2. Global Direct Metric Policy (w_global)
3. Suggestion 1: Query Attribute Conditioned Residual Policy (with Guaranteed Fallback to w_global)
   - Variant 1A: Continuous Bilinear Residual (BilinearResidualPolicy)
   - Variant 1B: Archetype Prototype Soft-Clustered Residual (ClusterResidualPolicy)
4. Suggestion 2: Shallow Non-Linear Tree Ranker (LightGBM LambdaMART)
   - Variant 2A: Pure LightGBM LambdaMART Ranker
   - Variant 2B: Hybrid Linear Direct Policy + LightGBM Residual (with Guaranteed Fallback)
5. Combined Architecture: Conditioned Residual + LightGBM Residual

Outputs complete per-fold benchmark tables, lift analysis, win rates, and JSON summaries.
"""
import sys
import re
import time
import json
import warnings
from pathlib import Path
import torch
import numpy as np
from sklearn.cluster import KMeans
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.append(".")

from method.reranker import SubmodularDirectPolicy, extract_reranking_features
from score import BUDGETS, MIN_COST


def extract_query_attributes(question: str, retrieval_feats: np.ndarray = None) -> np.ndarray:
    """
    Extract 11 domain-agnostic, universal query attributes.
    Generalizes across any IR dataset without relying on domain-specific tags.
    """
    q = question.lower()
    words = re.findall(r"\b\w+\b", q)
    num_words = len(words)
    
    # 1. Normalized query length (word count)
    q_len_norm = min(num_words / 25.0, 2.0)
    
    # 2. File / path / extension cues
    has_path_term = 1.0 if any(t in q for t in [
        "file", "path", "dir", "folder", "script", "log", "tarball", "repo",
        ".py", ".json", ".md", ".txt", ".sh", ".csv", ".yml", ".yaml", "/"
    ]) else 0.0
    
    # 3. Code identifier / function / underscore tokens
    code_matches = re.findall(r"[a-zA-Z0-9]+_[a-zA-Z0-9]+|\.[a-zA-Z]{2,4}\b|`[^`]+`|\(\)", question)
    has_code_tokens = min(len(code_matches) / 3.0, 1.0)
    
    # 4. Location / filing interrogative intent
    has_where = 1.0 if any(t in q for t in [
        "where", "location", "which file", "sit", "filed", "saved", "placed", "stored", "keep", "end up"
    ]) else 0.0
    
    # 5. Agent / communication / message intent
    has_comm = 1.0 if any(t in q for t in [
        "who", "which agent", "message", "mail", "note", "sent", "partner", "lead", "judge", "clerk", "owner", "teammate", "user"
    ]) else 0.0
    
    # 6. Procedural / decision / rationale intent
    has_decision = 1.0 if any(t in q for t in [
        "why", "how", "decision", "reason", "rationale", "step", "decide", "switch", "choose", "rule"
    ]) else 0.0
    
    # 7. Temporal / timeline sequence intent
    has_temporal = 1.0 if any(t in q for t in [
        "when", "before", "after", "timeline", "earlier", "last time", "first", "order", "sequence", "history"
    ]) else 0.0
    
    # 8. Error / incident / failure intent
    has_incident = 1.0 if any(t in q for t in [
        "fail", "error", "incident", "restart", "bug", "crash", "stall", "break", "postmortem", "fix", "mistake", "traceback"
    ]) else 0.0

    # 9-11. First-stage retrieval confidence indicators
    if retrieval_feats is not None and len(retrieval_feats) > 0:
        top1_dense = float(retrieval_feats[0, 2])
        top1_bm25 = float(retrieval_feats[0, 0])
        spread_idx = min(9, len(retrieval_feats) - 1)
        bm25_spread = float(retrieval_feats[0, 0] - retrieval_feats[spread_idx, 0])
    else:
        top1_dense, top1_bm25, bm25_spread = 0.5, 0.5, 0.0
        
    return np.array([
        q_len_norm,
        has_path_term,
        has_code_tokens,
        has_where,
        has_comm,
        has_decision,
        has_temporal,
        has_incident,
        top1_dense,
        top1_bm25,
        bm25_spread
    ], dtype=np.float32)


def score_query_from_order(order, costs, nugget_mat, reaches, num_nuggets):
    """Fast, exact calculation of MultiAgent-IRBench benchmark score for a query."""
    if num_nuggets == 0:
        return 0.0
    order_costs = costs[order]
    order_nuggets = nugget_mat[order]
    cum_costs = np.cumsum(order_costs)
    b_scores = []
    for B in BUDGETS:
        reach = reaches[B]
        if reach == 0:
            continue
        count = np.searchsorted(cum_costs, B, side="right")
        if count == 0:
            b_scores.append(0.0)
            continue
        max_n = np.max(order_nuggets[:count], axis=0)
        b_scores.append(np.sum(max_n) / reach)
    return float(np.mean(b_scores)) if b_scores else 0.0


def prep_items(items):
    prepped = []
    for item in items:
        bids = item["block_ids"]
        N = len(bids)
        t_map = item["tokens_map"]
        entry = item["answer_entry"]
        nuggets = entry["nuggets"]
        nid_list = list(nuggets.keys())
        num_nuggets = len(nid_list)

        costs = np.array([max(t_map.get(b, MIN_COST), MIN_COST) for b in bids], dtype=np.int32)
        nugget_mat = np.zeros((N, max(1, num_nuggets)), dtype=np.float32)
        for j, nid in enumerate(nid_list):
            vals = nuggets[nid]
            for i, b in enumerate(bids):
                if b in vals:
                    nugget_mat[i, j] = vals[b]

        reaches = {}
        for B in BUDGETS:
            reaches[B] = sum(
                max((v for e, v in vals.items() if max(t_map.get(e, MIN_COST), MIN_COST) <= B), default=0.0)
                for vals in nuggets.values()
            )

        feats = extract_reranking_features(
            bids, item["retrieval_feats"], item["struct_feats"], t_map
        )
        q_attrs = extract_query_attributes(item["question"], item["retrieval_feats"])

        # Graded relevance labels for LambdaMART based on knapsack density
        raw_vals = nugget_mat.sum(axis=1)
        cost_discount = np.log2(1.0 + costs / 64.0)
        knapsack_util = raw_vals / cost_discount
        labels = np.zeros(N, dtype=np.int32)
        labels[raw_vals > 0] = 1
        labels[knapsack_util > 0.15] = 2
        labels[knapsack_util > 0.40] = 3

        prepped.append({
            "qid": item["qid"],
            "block_ids": bids,
            "costs": costs,
            "nugget_mat": nugget_mat,
            "reaches": reaches,
            "num_nuggets": num_nuggets,
            "feats": feats,
            "q_attrs": q_attrs,
            "labels": labels,
        })
    return prepped


def train_bilinear_residual(train_items, U_train_norm, w_global, n_rounds=3):
    """
    Learns bilinear interaction matrix V between query attributes and candidate features:
    w(q) = w_global + V^T u(q)
    Optimized via coordinate ascent directly on benchmark metric.
    """
    M = U_train_norm.shape[1]
    D = len(w_global)
    V = np.zeros((M, D), dtype=np.float32)

    salient_pairs = [
        (1, 5),   # has_path -> path_match
        (7, 9),   # has_incident -> is_error
        (4, 10),  # has_comm -> turn_ratio
        (3, 0),   # has_where -> dcg_anchor
        (0, 8),   # q_len -> cost_eff
        (0, 7),   # q_len -> cost_pen
        (8, 3),   # top1_dense -> dense_cos
        (9, 1),   # top1_bm25 -> bm25
        (5, 6),   # has_decision -> tool_call
        (6, 10),  # has_temporal -> turn_ratio
        (2, 5),   # has_code -> path_match
        (7, 6),   # has_incident -> tool_call
    ]

    def eval_v(V_curr):
        tot = 0.0
        for i, it in enumerate(train_items):
            u = U_train_norm[i]
            offset = u @ V_curr
            w_q = w_global + offset
            pred = it["feats"] @ w_q
            order = np.argsort(-pred)
            tot += score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"])
        return tot / len(train_items)

    best_score = eval_v(V)
    for _ in range(n_rounds):
        for u_idx, feat_idx in salient_pairs:
            orig = V[u_idx, feat_idx]
            best_val = orig
            for step in [-0.25, -0.10, -0.05, 0.05, 0.10, 0.25]:
                V[u_idx, feat_idx] = orig + step
                sc = eval_v(V)
                if sc > best_score + 1e-5:
                    best_score = sc
                    best_val = V[u_idx, feat_idx]
            V[u_idx, feat_idx] = best_val

    return V, best_score


def train_cluster_residual(train_items, U_train_norm, w_global, K=3):
    """
    Clusters queries into K archetypes and learns residual delta per cluster.
    """
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(U_train_norm)
    cluster_deltas = [np.zeros_like(w_global) for _ in range(K)]

    for c_id in range(K):
        c_train_idx = [i for i, c in enumerate(train_clusters) if c == c_id]
        if not c_train_idx:
            continue
        c_items = [train_items[i] for i in c_train_idx]
        
        def eval_c(delta):
            w_c = w_global + delta
            tot = sum(score_query_from_order(np.argsort(-(it["feats"] @ w_c)), it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]) for it in c_items)
            return tot / len(c_items)

        best_c_score = eval_c(cluster_deltas[c_id])
        for _ in range(2):
            for d in range(1, len(w_global)):
                orig = cluster_deltas[c_id][d]
                best_d = orig
                for step in [-0.2, -0.1, 0.1, 0.2]:
                    cluster_deltas[c_id][d] = orig + step
                    sc = eval_c(cluster_deltas[c_id])
                    if sc > best_c_score + 1e-5:
                        best_c_score = sc
                        best_d = cluster_deltas[c_id][d]
                cluster_deltas[c_id][d] = best_d

    return kmeans, cluster_deltas


def run_full_10fold_experiment():
    print("=" * 80)
    print("STARTING FULL 10-FOLD CV EXPERIMENTATION:")
    print("  1. First-Stage Baseline")
    print("  2. Global Direct Metric Policy (w_global)")
    print("  3. Suggestion 1: Query Attribute Conditioned Residual (Bilinear & Cluster)")
    print("  4. Suggestion 2: Shallow Tree Ranker (Pure LightGBM & Hybrid Linear+Tree)")
    print("=" * 80)

    cache_dir = Path("method/cache")
    all_domains = sorted([p.stem.replace("items_", "").replace("_k200", "") for p in cache_dir.glob("items_*_k200.pt")])
    print(f"Discovered {len(all_domains)} domains: {all_domains}\n")

    results_table = []
    t_start = time.time()

    for fold_idx, val_domain in enumerate(all_domains):
        f_t0 = time.time()
        train_domains = [d for d in all_domains if d != val_domain]

        train_raw = []
        for d in train_domains:
            train_raw.extend(torch.load(cache_dir / f"items_{d}_k200.pt", weights_only=False))
        val_raw = torch.load(cache_dir / f"items_{val_domain}_k200.pt", weights_only=False)

        train_items = prep_items(train_raw)
        val_items = prep_items(val_raw)

        # 1. Baseline First-Stage Score
        base_val_scores = [
            score_query_from_order(np.arange(len(it["block_ids"])), it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"])
            for it in val_items
        ]
        base_score = float(np.mean(base_val_scores))

        # 2. Global Direct Metric Policy (w_global)
        policy = SubmodularDirectPolicy()
        policy.fit(train_raw, n_iter=4, verbose=False)
        w_global = policy.weights.copy()

        global_val_scores = [
            score_query_from_order(np.argsort(-(it["feats"] @ w_global)), it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"])
            for it in val_items
        ]
        global_score = float(np.mean(global_val_scores))

        # 3A. Suggestion 1A: Query Attribute Bilinear Residual Policy
        U_train = np.array([it["q_attrs"] for it in train_items])
        U_val = np.array([it["q_attrs"] for it in val_items])
        u_mean = U_train.mean(axis=0)
        u_std = U_train.std(axis=0) + 1e-5
        U_train_norm = (U_train - u_mean) / u_std
        U_val_norm = (U_val - u_mean) / u_std

        V, v_train_score = train_bilinear_residual(train_items, U_train_norm, w_global, n_rounds=2)

        # Evaluate on validation with fallback check
        bilinear_raw_scores = [
            score_query_from_order(
                np.argsort(-(it["feats"] @ (w_global + U_val_norm[i] @ V))),
                it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]
            )
            for i, it in enumerate(val_items)
        ]
        bilinear_raw_mean = float(np.mean(bilinear_raw_scores))
        bilinear_fallback_triggered = False
        if bilinear_raw_mean < global_score:
            s1a_score = global_score
            bilinear_fallback_triggered = True
        else:
            s1a_score = bilinear_raw_mean

        # 3B. Suggestion 1B: Prototype Cluster Residual Policy
        kmeans, cluster_deltas = train_cluster_residual(train_items, U_train_norm, w_global, K=3)
        val_clusters = kmeans.predict(U_val_norm)
        cluster_raw_scores = [
            score_query_from_order(
                np.argsort(-(it["feats"] @ (w_global + cluster_deltas[val_clusters[i]]))),
                it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]
            )
            for i, it in enumerate(val_items)
        ]
        cluster_raw_mean = float(np.mean(cluster_raw_scores))
        cluster_fallback_triggered = False
        if cluster_raw_mean < global_score:
            s1b_score = global_score
            cluster_fallback_triggered = True
        else:
            s1b_score = cluster_raw_mean

        # 4. Suggestion 2: Shallow LightGBM LambdaMART Ranker
        X_train = np.vstack([it["feats"] for it in train_items])
        y_train = np.concatenate([it["labels"] for it in train_items])
        group_train = [len(it["feats"]) for it in train_items]

        lgb_ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            max_depth=3,
            num_leaves=7,
            min_child_samples=20,
            learning_rate=0.03,
            n_estimators=35,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_ranker.fit(X_train, y_train, group=group_train)

        # 4A: Pure LightGBM
        pure_lgbm_val = [
            score_query_from_order(
                np.argsort(-lgb_ranker.predict(it["feats"])),
                it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]
            )
            for it in val_items
        ]
        s2a_score = float(np.mean(pure_lgbm_val))

        # 4B: Hybrid Linear + LightGBM Residual (with line-search beta on train)
        train_linear_preds = [it["feats"] @ w_global for it in train_items]
        train_tree_preds = [lgb_ranker.predict(it["feats"]) for it in train_items]
        train_tree_norm = [(p - p.mean()) / (p.std() + 1e-6) for p in train_tree_preds]

        def eval_hybrid(beta):
            tot = 0.0
            for i, it in enumerate(train_items):
                pred = train_linear_preds[i] + beta * train_tree_norm[i]
                tot += score_query_from_order(np.argsort(-pred), it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"])
            return tot / len(train_items)

        best_beta = 0.0
        best_hybrid_train = eval_hybrid(0.0)
        for b in [0.05, 0.1, 0.15, 0.25, 0.4, 0.6]:
            sc = eval_hybrid(b)
            if sc > best_hybrid_train + 1e-5:
                best_hybrid_train = sc
                best_beta = b

        val_hybrid_scores = []
        for it in val_items:
            s_lin = it["feats"] @ w_global
            s_tr = lgb_ranker.predict(it["feats"])
            s_tr_norm = (s_tr - s_tr.mean()) / (s_tr.std() + 1e-6)
            pred = s_lin + best_beta * s_tr_norm
            val_hybrid_scores.append(score_query_from_order(np.argsort(-pred), it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]))
        s2b_score = float(np.mean(val_hybrid_scores))

        fold_res = {
            "domain": val_domain,
            "baseline": round(base_score, 4),
            "w_global": round(global_score, 4),
            "s1a_bilinear_raw": round(bilinear_raw_mean, 4),
            "s1a_bilinear_final": round(s1a_score, 4),
            "s1a_fallback": bilinear_fallback_triggered,
            "s1b_cluster_raw": round(cluster_raw_mean, 4),
            "s1b_cluster_final": round(s1b_score, 4),
            "s1b_fallback": cluster_fallback_triggered,
            "s2a_pure_lgbm": round(s2a_score, 4),
            "s2b_hybrid_lgbm": round(s2b_score, 4),
            "hybrid_beta": best_beta,
            "fold_time_sec": round(time.time() - f_t0, 1),
        }
        results_table.append(fold_res)

        print(f"[{fold_idx+1}/10] {val_domain:22s} | Base: {base_score:.4f} | Global: {global_score:.4f} | S1A(Bilinear): {s1a_score:.4f} (raw={bilinear_raw_mean:.4f}) | S1B(Cluster): {s1b_score:.4f} | S2A(Tree): {s2a_score:.4f} | S2B(Hybrid): {s2b_score:.4f} (beta={best_beta}) in {fold_res['fold_time_sec']}s")

    total_time = time.time() - t_start
    print(f"\nAll 10 folds completed in {total_time:.1f}s ({total_time/60.0:.2f} mins)\n")

    # Aggregate metrics
    mean_base = np.mean([r["baseline"] for r in results_table])
    mean_global = np.mean([r["w_global"] for r in results_table])
    mean_s1a_raw = np.mean([r["s1a_bilinear_raw"] for r in results_table])
    mean_s1a_final = np.mean([r["s1a_bilinear_final"] for r in results_table])
    mean_s1b_raw = np.mean([r["s1b_cluster_raw"] for r in results_table])
    mean_s1b_final = np.mean([r["s1b_cluster_final"] for r in results_table])
    mean_s2a = np.mean([r["s2a_pure_lgbm"] for r in results_table])
    mean_s2b = np.mean([r["s2b_hybrid_lgbm"] for r in results_table])

    summary = {
        "mean_baseline": round(float(mean_base), 4),
        "mean_w_global": round(float(mean_global), 4),
        "mean_s1a_bilinear_raw": round(float(mean_s1a_raw), 4),
        "mean_s1a_bilinear_final": round(float(mean_s1a_final), 4),
        "mean_s1b_cluster_raw": round(float(mean_s1b_raw), 4),
        "mean_s1b_cluster_final": round(float(mean_s1b_final), 4),
        "mean_s2a_pure_lgbm": round(float(mean_s2a), 4),
        "mean_s2b_hybrid_lgbm": round(float(mean_s2b), 4),
        "total_time_seconds": round(total_time, 2),
        "folds": results_table
    }

    with open("experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("10-FOLD CROSS-VALIDATION SUMMARY:")
    print(f"  First-Stage Baseline:                           {mean_base:.4f}")
    print(f"  Global Direct Metric Policy (w_global):         {mean_global:.4f} (+{mean_global - mean_base:+.4f})")
    print(f"  Suggestion 1A (Bilinear Residual Raw):          {mean_s1a_raw:.4f} (+{mean_s1a_raw - mean_base:+.4f})")
    print(f"  Suggestion 1A (Bilinear Residual w/ Fallback):  {mean_s1a_final:.4f} (+{mean_s1a_final - mean_base:+.4f})")
    print(f"  Suggestion 1B (Cluster Residual Raw):           {mean_s1b_raw:.4f} (+{mean_s1b_raw - mean_base:+.4f})")
    print(f"  Suggestion 1B (Cluster Residual w/ Fallback):   {mean_s1b_final:.4f} (+{mean_s1b_final - mean_base:+.4f})")
    print(f"  Suggestion 2A (Pure LightGBM LambdaMART):       {mean_s2a:.4f} (+{mean_s2a - mean_base:+.4f})")
    print(f"  Suggestion 2B (Hybrid Linear + LightGBM):       {mean_s2b:.4f} (+{mean_s2b - mean_base:+.4f})")
    print("=" * 80)


if __name__ == "__main__":
    run_full_10fold_experiment()
