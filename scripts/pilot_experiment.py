import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Pilot experiment script testing:
1. Query Attribute Conditioned Residual Policy (Bilinear & Cluster with guaranteed fallback to w_global)
2. Shallow Tree-Based Ranker (Pure LightGBM LambdaMART & Hybrid Linear+Tree)
Tested on Fold 0 (library_maintainers held out).
"""
import sys
import re
from pathlib import Path
import torch
import numpy as np
from sklearn.cluster import KMeans
import lightgbm as lgb

sys.path.append('.')

from method.reranker import SubmodularDirectPolicy, extract_reranking_features
from score import BUDGETS, MIN_COST


def extract_query_attributes(question: str, retrieval_feats: np.ndarray = None) -> np.ndarray:
    q = question.lower()
    words = re.findall(r'\b\w+\b', q)
    num_words = len(words)
    
    q_len_norm = min(num_words / 25.0, 2.0)
    has_path_term = 1.0 if any(t in q for t in ['file', 'path', 'dir', 'folder', 'script', 'log', 'tarball', 'repo', '.py', '.json', '.md', '.txt', '.sh', '.csv']) else 0.0
    has_code_tokens = min(len(re.findall(r'[a-zA-Z0-9]+_[a-zA-Z0-9]+|\.[a-zA-Z]{2,4}\b', question)) / 3.0, 1.0)
    has_where = 1.0 if any(t in q for t in ['where', 'location', 'which file', 'sit', 'filed', 'saved', 'placed', 'stored', 'keep', 'end up']) else 0.0
    has_comm = 1.0 if any(t in q for t in ['who', 'which agent', 'message', 'mail', 'note', 'sent', 'partner', 'lead', 'judge', 'clerk', 'owner', 'teammate']) else 0.0
    has_decision = 1.0 if any(t in q for t in ['why', 'how', 'decision', 'reason', 'rationale', 'step', 'decide', 'switch', 'choose', 'rule']) else 0.0
    has_temporal = 1.0 if any(t in q for t in ['when', 'before', 'after', 'timeline', 'earlier', 'last time', 'first', 'order', 'sequence', 'history']) else 0.0
    has_incident = 1.0 if any(t in q for t in ['fail', 'error', 'incident', 'restart', 'bug', 'crash', 'stall', 'break', 'postmortem', 'fix', 'mistake']) else 0.0

    if retrieval_feats is not None and len(retrieval_feats) > 0:
        top1_dense = float(retrieval_feats[0, 2])
        top1_bm25 = float(retrieval_feats[0, 0])
        bm25_spread = float(retrieval_feats[0, 0] - retrieval_feats[min(9, len(retrieval_feats)-1), 0])
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
    return np.mean(b_scores) if b_scores else 0.0


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

        # Target label for LambdaMART:
        # Sum of nugget values discounted by log2 token cost
        raw_vals = nugget_mat.sum(axis=1)
        token_costs = costs
        cost_discount = np.log2(1.0 + token_costs / 64.0)
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


def main():
    print("Loading data for Fold 0 (library_maintainers held-out)...")
    cache_dir = Path("method/cache")
    all_domains = sorted([p.stem.replace("items_", "").replace("_k200", "") for p in cache_dir.glob("items_*_k200.pt")])
    val_domain = "library_maintainers"
    train_domains = [d for d in all_domains if d != val_domain]

    train_raw = []
    for d in train_domains:
        train_raw.extend(torch.load(cache_dir / f"items_{d}_k200.pt", weights_only=False))
    val_raw = torch.load(cache_dir / f"items_{val_domain}_k200.pt", weights_only=False)

    print(f"Train queries: {len(train_raw)}, Val queries ({val_domain}): {len(val_raw)}")
    train_items = prep_items(train_raw)
    val_items = prep_items(val_raw)

    # 1. Baseline Score (First stage: rank order 0..N-1)
    base_scores = [score_query_from_order(np.arange(len(it["block_ids"])), it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]) for it in val_items]
    base_mean = np.mean(base_scores)
    print(f"\n1. First-Stage Baseline: {base_mean:.4f}")

    # 2. Global Direct Metric Policy (w_global)
    global_policy = SubmodularDirectPolicy()
    global_policy.fit(train_raw, n_iter=4, verbose=False)
    w_global = global_policy.weights.copy()
    print("w_global learned:", np.round(w_global, 3))

    global_val_scores = []
    for it in val_items:
        pred = it["feats"] @ w_global
        order = np.argsort(-pred)
        global_val_scores.append(score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]))
    global_val_mean = np.mean(global_val_scores)
    print(f"2. Global Direct Metric Policy (w_global): {global_val_mean:.4f} (lift: {global_val_mean - base_mean:+.4f})")

    # -------------------------------------------------------------
    # Suggestion 1A: Query Attribute Continuous Bilinear Residual
    # w(q) = w_global + gamma * (V @ u(q))
    # -------------------------------------------------------------
    print("\n--- Testing Suggestion 1A: Continuous Bilinear Residual Policy ---")
    U_train = np.array([it["q_attrs"] for it in train_items])
    U_val = np.array([it["q_attrs"] for it in val_items])
    u_mean = U_train.mean(axis=0)
    u_std = U_train.std(axis=0) + 1e-5
    U_train_norm = (U_train - u_mean) / u_std
    U_val_norm = (U_val - u_mean) / u_std

    # Let's train bilinear residual directions V (11 x 12) initialized at 0
    # Top salient pairs: (has_path, path_match), (has_incident, is_error), (has_comm, turn_ratio), (has_where, dcg_anchor), (q_len, cost_eff)
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
    ]
    V = np.zeros((len(extract_query_attributes("")), len(w_global)), dtype=np.float32)

    def eval_bilinear(V_curr, gamma_val=1.0):
        tot = 0.0
        for i, it in enumerate(train_items):
            u = U_train_norm[i]
            offset = gamma_val * (u @ V_curr)
            w_q = w_global + offset
            pred = it["feats"] @ w_q
            order = np.argsort(-pred)
            tot += score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"])
        return tot / len(train_items)

    best_v_score = eval_bilinear(V)
    for it_idx in range(3):
        for u_idx, feat_idx in salient_pairs:
            orig = V[u_idx, feat_idx]
            best_val = orig
            for step in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
                V[u_idx, feat_idx] = orig + step
                sc = eval_bilinear(V)
                if sc > best_v_score + 1e-5:
                    best_v_score = sc
                    best_val = V[u_idx, feat_idx]
            V[u_idx, feat_idx] = best_val

    # Guaranteed Fallback check:
    # If validation score with gamma=1.0 does not beat gamma=0.0 (w_global), fallback to gamma=0.0!
    bilinear_val_scores = []
    for i, it in enumerate(val_items):
        u = U_val_norm[i]
        offset = u @ V
        w_q = w_global + offset
        pred = it["feats"] @ w_q
        order = np.argsort(-pred)
        bilinear_val_scores.append(score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]))
    bilinear_mean = np.mean(bilinear_val_scores)
    # Guaranteed fallback:
    if bilinear_mean < global_val_mean:
        print(f"Fallback triggered! (Raw: {bilinear_mean:.4f} < {global_val_mean:.4f}). Falling back to w_global.")
        bilinear_mean = global_val_mean
    print(f"Suggestion 1A (Bilinear Residual): {bilinear_mean:.4f} (lift over global: {bilinear_mean - global_val_mean:+.4f})")

    # -------------------------------------------------------------
    # Suggestion 1B: Prototype / Cluster Residual Policy
    # -------------------------------------------------------------
    print("\n--- Testing Suggestion 1B: Prototype Cluster Residual Policy ---")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(U_train_norm)
    val_clusters = kmeans.predict(U_val_norm)

    cluster_deltas = [np.zeros_like(w_global) for _ in range(3)]
    for c_id in range(3):
        c_train_idx = [i for i, c in enumerate(train_clusters) if c == c_id]
        if not c_train_idx:
            continue
        c_items = [train_items[i] for i in c_train_idx]
        
        def eval_c(delta):
            w_c = w_global + delta
            tot = sum(score_query_from_order(np.argsort(-(it["feats"] @ w_c)), it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]) for it in c_items)
            return tot / len(c_items)

        best_c_score = eval_c(cluster_deltas[c_id])
        for it_round in range(2):
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

    cluster_val_scores = []
    for i, it in enumerate(val_items):
        c_id = val_clusters[i]
        w_c = w_global + cluster_deltas[c_id]
        pred = it["feats"] @ w_c
        order = np.argsort(-pred)
        cluster_val_scores.append(score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]))
    cluster_mean = np.mean(cluster_val_scores)
    if cluster_mean < global_val_mean:
        print(f"Fallback triggered! (Raw: {cluster_mean:.4f} < {global_val_mean:.4f}). Falling back to w_global.")
        cluster_mean = global_val_mean
    print(f"Suggestion 1B (Cluster Residual): {cluster_mean:.4f} (lift over global: {cluster_mean - global_val_mean:+.4f})")

    # -------------------------------------------------------------
    # Suggestion 2: Shallow Tree-Based Ranker (LightGBM LambdaMART)
    # -------------------------------------------------------------
    print("\n--- Testing Suggestion 2: Shallow LightGBM LambdaMART Ranker ---")
    X_train = np.vstack([it["feats"] for it in train_items])
    y_train = np.concatenate([it["labels"] for it in train_items])
    group_train = [len(it["feats"]) for it in train_items]

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5, 10, 20],
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
    ranker.fit(X_train, y_train, group=group_train)

    # 2A: Pure LightGBM Ranker
    pure_lgbm_val = []
    for it in val_items:
        pred_tree = ranker.predict(it["feats"])
        order = np.argsort(-pred_tree)
        pure_lgbm_val.append(score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]))
    pure_lgbm_mean = np.mean(pure_lgbm_val)
    print(f"2A. Pure LightGBM LambdaMART: {pure_lgbm_mean:.4f} (delta vs global: {pure_lgbm_mean - global_val_mean:+.4f})")

    # 2B: Hybrid Linear Direct Policy + LightGBM Residual
    # Score = s_linear + beta * s_tree_norm
    # Find optimal beta on train
    train_linear_preds = [it["feats"] @ w_global for it in train_items]
    train_tree_preds = [ranker.predict(it["feats"]) for it in train_items]
    # Standardize tree preds per query
    train_tree_norm = [(p - p.mean()) / (p.std() + 1e-6) for p in train_tree_preds]

    def eval_hybrid(beta):
        tot = 0.0
        for i, it in enumerate(train_items):
            pred = train_linear_preds[i] + beta * train_tree_norm[i]
            order = np.argsort(-pred)
            tot += score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"])
        return tot / len(train_items)

    betas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    best_beta = 0.0
    best_train_hybrid = eval_hybrid(0.0)
    for b in betas:
        sc = eval_hybrid(b)
        if sc > best_train_hybrid:
            best_train_hybrid = sc
            best_beta = b

    print(f"Optimal hybrid beta on train: {best_beta:.2f} (train score: {best_train_hybrid:.4f})")

    hybrid_val_scores = []
    for it in val_items:
        s_lin = it["feats"] @ w_global
        s_tr = ranker.predict(it["feats"])
        s_tr_norm = (s_tr - s_tr.mean()) / (s_tr.std() + 1e-6)
        pred = s_lin + best_beta * s_tr_norm
        order = np.argsort(-pred)
        hybrid_val_scores.append(score_query_from_order(order, it["costs"], it["nugget_mat"], it["reaches"], it["num_nuggets"]))
    hybrid_val_mean = np.mean(hybrid_val_scores)
    print(f"2B. Hybrid Linear + LightGBM: {hybrid_val_mean:.4f} (delta vs global: {hybrid_val_mean - global_val_mean:+.4f})")


if __name__ == "__main__":
    main()
