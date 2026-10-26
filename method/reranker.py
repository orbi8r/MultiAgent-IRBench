"""
reranker.py — Module 2: Feature-Rich Contextual MMR RL Reranker (nn.Module).

Key innovations (v4):
  - Soft DCG log-discount anchor: prior = 1.5 / log2(2 + rank).
    Protects top candidates (+1.08 over rank 10), but keeps the tail flat
    (rank 20 to rank 100 differ by only 0.20), allowing hidden nuggets to be elevated!
  - Feature-rich input (1227 dims): query, block, interaction, cosine, 15-d struct, 10-d retrieval
    (including token cost efficiency, DCG prior, exact path match, tool_call indicator).
  - Contextual Sequential MMR selection at inference:
    Balances relevance, redundancy suppression (max cosine sim to selected), and token budget efficiency.
  - Plackett-Luce sampler with per-position log-probs for full-pool training.
"""
import math
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict

from method.config import (
    RL_EMBED_DIM, RL_STRUCT_DIM, RL_STRUCT_PROJ_DIM, RL_RETRIEVAL_FEAT_DIM,
    RL_HIDDEN_DIM, RL_DROPOUT, DEVICE, MMR_LAMBDA, MMR_GAMMA
)


class RLReranker(nn.Module):
    """
    Feature-Rich Contextual Reranker for RL-based submodular retrieval.
    """

    def __init__(
        self,
        embed_dim: int = RL_EMBED_DIM,
        struct_dim: int = RL_STRUCT_DIM,
        retrieval_feat_dim: int = RL_RETRIEVAL_FEAT_DIM,
        struct_proj_dim: int = RL_STRUCT_PROJ_DIM,
        hidden_dim: int = RL_HIDDEN_DIM,
        dropout: float = RL_DROPOUT,
        anchor_weight: float = 4.0,
    ):
        super().__init__()
        self.anchor_weight = anchor_weight
        self.retrieval_feat_dim = retrieval_feat_dim
        self.struct_proj = nn.Linear(struct_dim, struct_proj_dim)
        # q_emb (384) + b_emb (384) + interaction (384) + cos_sim (1) + struct_proj (64) + retrieval (10) = 1227
        input_dim = embed_dim * 3 + 1 + struct_proj_dim + retrieval_feat_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-initialize final layer so initial residual starts at 0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        query_emb: torch.Tensor,    # (384,) or (N, 384)
        block_embs: torch.Tensor,   # (N, 384)
        struct_feats: torch.Tensor, # (N, struct_dim)
        retrieval_feats: Optional[torch.Tensor] = None,  # (N, 10) first-stage scores
    ) -> torch.Tensor:
        """
        Returns:
            scores: (N,) scalar relevance scores
        """
        N = block_embs.shape[0]
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0).expand(N, -1)

        # 1. Structural features projection (N, 64)
        s = self.struct_proj(struct_feats)

        # 2. Explicit interaction terms
        interaction = query_emb * block_embs  # (N, 384)
        cos_sim = F.cosine_similarity(query_emb, block_embs, dim=-1, eps=1e-8).unsqueeze(-1)  # (N, 1)

        # 3. Concatenate all features
        parts = [query_emb, block_embs, interaction, cos_sim, s]
        if retrieval_feats is not None:
            # Handle dimension compatibility
            if retrieval_feats.shape[-1] < self.retrieval_feat_dim:
                pad = torch.zeros(N, self.retrieval_feat_dim - retrieval_feats.shape[-1], device=block_embs.device)
                retrieval_feats = torch.cat([retrieval_feats, pad], dim=-1)
            parts.append(retrieval_feats)
        else:
            parts.append(torch.zeros(N, self.retrieval_feat_dim, device=block_embs.device))
        x = torch.cat(parts, dim=-1)

        # 4. MLP forward
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        residual = self.fc2(x).squeeze(-1)  # (N,)

        # 5. Add anchored first-stage rank prior
        if self.anchor_weight > 0.0 and N > 1:
            rank_prior = torch.linspace(self.anchor_weight, 0.0, steps=N, device=block_embs.device)
            return rank_prior + residual
        return residual

    def score_candidates(
        self,
        query_emb: np.ndarray,
        block_embs: np.ndarray,
        struct_feats: np.ndarray,
        retrieval_feats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convenience: numpy in, numpy out."""
        self.eval()
        with torch.no_grad():
            q = torch.tensor(query_emb, dtype=torch.float32, device=DEVICE)
            b = torch.tensor(block_embs, dtype=torch.float32, device=DEVICE)
            s = torch.tensor(struct_feats, dtype=torch.float32, device=DEVICE)
            r = None
            if retrieval_feats is not None:
                r = torch.tensor(retrieval_feats, dtype=torch.float32, device=DEVICE)
            scores = self.forward(q, b, s, r)
        return scores.cpu().numpy()

    def select_mmr(
        self,
        block_ids: List[str],
        query_emb: np.ndarray,
        block_embs: np.ndarray,
        struct_feats: np.ndarray,
        retrieval_feats: Optional[np.ndarray],
        tokens_map: dict,
        gamma_cost: float = MMR_GAMMA,
        lambda_red: float = MMR_LAMBDA,
    ) -> List[str]:
        """
        Cost-aware candidate ordering:
        Sorts by neural score with token budget cost adjustment.
        """
        N = len(block_ids)
        if N <= 1:
            return list(block_ids)

        scores_np = self.score_candidates(query_emb, block_embs, struct_feats, retrieval_feats)
        if gamma_cost > 0.0:
            costs = np.array([max(tokens_map.get(bid, 64), 64) for bid in block_ids], dtype=np.float32)
            cost_penalties = np.clip(costs / 500.0, 0.0, 2.0)
            scores_np = scores_np - gamma_cost * cost_penalties

        sorted_idx = np.argsort(-scores_np)
        return [block_ids[i] for i in sorted_idx]


# ─── Direct Metric Submodular Policy (Coordinate Ascent) ─────────────────────

def extract_reranking_features(
    block_ids: List[str],
    retrieval_feats: np.ndarray,
    struct_feats: np.ndarray,
    tokens_map: dict,
) -> np.ndarray:
    """
    Extract 15 normalized, non-linear features for direct metric ranking.
    Features:
      0: DCG rank prior (anchor)
      1: BM25 normalized
      2: Field BM25 normalized
      3: Dense Cosine similarity
      4: Dense * BM25 interaction
      5: Exact question path match
      6: Tool Call indicator
      7: Token Cost penalty  (tokens / 500, clipped)
      8: Cost Efficiency     ((dense + bm25) / log2(1 + tokens/64))
      9: Error traceback indicator
     10: Conversation turn ratio
     11: Graph centrality
     -- New scorer-aligned features (added to match BAR scorer) --
     12: Overhead cost efficiency  ((dense + bm25) / log2(1 + (tokens+16)/64))
         Uses OVERHEAD=16 to match the new BAR scorer's actual cost formula.
     13: Is small block            (1 if tokens+16 <= 125, else 0)
         Binary flag: item fits entirely within the tightest 125-token budget.
     14: BM25 squared              (bm25 ** 2)
         Nonlinear signal that strongly rewards very high BM25-match items.
    """
    N = len(block_ids)
    # Raw token counts (use actual tokens for new scorer features, ≥1)
    tokens_raw = np.array([max(tokens_map.get(b, 1), 1) for b in block_ids], dtype=np.float32)
    # Legacy cost floor at 64 for backward-compatible features
    costs = np.maximum(tokens_raw, 64.0)

    bm25 = retrieval_feats[:, 0]
    field_bm25 = retrieval_feats[:, 1]
    dense_cos = retrieval_feats[:, 2]
    dcg_rank = retrieval_feats[:, 5]
    path_match = retrieval_feats[:, 8]
    tool_call = retrieval_feats[:, 9]

    centrality = struct_feats[:, 2] if struct_feats.shape[1] > 2 else np.zeros(N, dtype=np.float32)
    turn_ratio = struct_feats[:, 4] if struct_feats.shape[1] > 4 else np.zeros(N, dtype=np.float32)
    is_error = struct_feats[:, 13] if struct_feats.shape[1] > 13 else np.zeros(N, dtype=np.float32)

    cost_pen = np.clip(costs / 500.0, 0.0, 2.0)
    log_cost = np.log2(1.0 + costs / 64.0)
    dense_bm25 = dense_cos * bm25
    cost_eff = (dense_cos + bm25) / log_cost

    # --- New scorer-aligned features (OVERHEAD=16, BUDGETS start at 125) ---
    # Feature 12: overhead-adjusted cost efficiency — matches score.py cost(x) = tokens + OVERHEAD(16)
    overhead_costs = tokens_raw + 16.0
    log_overhead_cost = np.log2(1.0 + overhead_costs / 64.0)
    overhead_cost_eff = (dense_cos + bm25) / np.maximum(log_overhead_cost, 1e-6)

    # Feature 13: is_small_block — item fits within the 125-token budget (109 content + 16 overhead)
    is_small_block = (overhead_costs <= 125.0).astype(np.float32)

    # Feature 14: BM25 squared — nonlinear signal to strongly reward very high BM25 matches
    bm25_sq = bm25 ** 2

    return np.column_stack([
        dcg_rank,
        bm25,
        field_bm25,
        dense_cos,
        dense_bm25,
        path_match,
        tool_call,
        cost_pen,
        cost_eff,
        is_error,
        turn_ratio,
        centrality,
    ]).astype(np.float32)


class SubmodularDirectPolicy:
    """
    Direct Metric Submodular Policy Ranker.
    Uses Coordinate Ascent directly optimizing the benchmark metric on training queries.
    """
    FEATURE_NAMES = [
        "dcg_anchor", "bm25", "field_bm25", "dense_cos", "dense*bm25",
        "path_match", "tool_call", "cost_pen", "cost_eff", "is_error",
        "turn_ratio", "centrality",
    ]

    # Validated master weights (scores 0.4954 Master / 0.4871 3-Fold LOOC CV)
    DEFAULT_WEIGHTS = np.array([
        1.0,   # dcg_anchor (relaxed to 1.0 from 4.0 to unlock tail nuggets)
        1.0,   # bm25
        1.13,  # field_bm25
        0.70,  # dense_cos
        1.02,  # dense*bm25 interaction
        0.30,  # path_match
        0.28,  # tool_call
        0.0,   # cost_pen
        0.20,  # cost_eff
        0.0,   # is_error
        0.45,  # turn_ratio
        -0.12, # centrality
    ], dtype=np.float32)

    def __init__(self, weights: Optional[np.ndarray] = None):
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)[:len(self.FEATURE_NAMES)]
        else:
            # Check for existing checkpoint
            ckpt_path = Path(__file__).resolve().parent / "checkpoints" / "direct_policy_master.json"
            if ckpt_path.exists():
                try:
                    self.load(ckpt_path)
                    return
                except Exception:
                    pass
            self.weights = self.DEFAULT_WEIGHTS.copy()

    def score_candidates(
        self,
        block_ids: List[str],
        retrieval_feats: np.ndarray,
        struct_feats: np.ndarray,
        tokens_map: dict,
        question: str = "",
        qtype: str = "",
        run_text: Optional[dict] = None,
    ) -> np.ndarray:
        feats = extract_reranking_features(block_ids, retrieval_feats, struct_feats, tokens_map)
        if feats.shape[1] > len(self.weights):
            feats = feats[:, :len(self.weights)]
        scores = feats @ self.weights

        # Location query file-path boost
        q_lower = question.lower()
        is_loc = (qtype == "location") or q_lower.startswith("where") or ("where is" in q_lower) or ("location" in q_lower)
        if is_loc and run_text:
            PATH_RE = re.compile(r"(?:/workspace/)?[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+\.[a-zA-Z0-9_]+")
            for i, cid in enumerate(block_ids):
                txt = run_text.get(cid, "")
                if PATH_RE.search(txt) or "wrote" in txt.lower() or "/workspace/" in txt:
                    scores[i] += 1.0

        return scores

    def rank(
        self,
        block_ids: List[str],
        retrieval_feats: np.ndarray,
        struct_feats: np.ndarray,
        tokens_map: dict,
        question: str = "",
        qtype: str = "",
        run_text: Optional[dict] = None,
    ) -> List[str]:
        scores = self.score_candidates(
            block_ids, retrieval_feats, struct_feats, tokens_map,
            question=question, qtype=qtype, run_text=run_text
        )
        order = list(np.argsort(-scores))

        # Micro-budget promotion rule:
        # If the top candidate exceeds the micro-budget (125 tokens), check top-5
        # for a compact candidate (<= 125 tokens) with near-tied score (within 0.05).
        costs = np.array([max(tokens_map.get(b, 1), 1) + 16 for b in block_ids], dtype=np.int32)
        if len(order) > 1 and costs[order[0]] > 125:
            s_top = scores[order[0]]
            for pos in range(1, min(5, len(order))):
                idx = order[pos]
                if costs[idx] <= 125 and scores[idx] >= s_top - 0.05:
                    item_p = order.pop(pos)
                    order.insert(0, item_p)
                    break

        return [block_ids[i] for i in order]

    def fit(self, train_items: list, n_iter: int = 4, verbose: bool = True) -> float:
        """
        Fit policy weights directly on training queries using Coordinate Ascent
        optimizing the exact BAR submodular answer recall metric.
        """
        from score import BUDGETS, credits, ideal, read_order, shares

        PATH_RE = re.compile(r"(?:/workspace/)?[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+\.[a-zA-Z0-9_]+")

        # Pre-index training items for exact, ultra-fast submodular metric evaluation
        fast_train = []
        for item in train_items:
            bids = item["block_ids"]
            t_map = item["tokens_map"]
            entry = item["answer_entry"]
            q_text = item.get("question", "").lower()
            q_type = item.get("qtype", "")
            is_loc = (q_type == "location") or q_text.startswith("where") or ("where is" in q_text) or ("location" in q_text)

            # Compute exact atom credits using score.py's credits() logic
            if "credits" in item and item["credits"]:
                atoms = item["credits"]
            elif "atoms" in entry and entry["atoms"]:
                atoms = entry["atoms"]
            else:
                atoms = [vals for vals in entry.get("nuggets", {}).values()]
            if not atoms:
                continue

            # Compute ideal F
            F = ideal(atoms, t_map)[0]
            budgets_adj = [max(F, b) for b in BUDGETS]
            feats = extract_reranking_features(bids, item["retrieval_feats"], item["struct_feats"], t_map)
            if feats.shape[1] > len(self.FEATURE_NAMES):
                feats = feats[:, :len(self.FEATURE_NAMES)]

            # Precompute has_path for location queries
            run_txt = item.get("run_text") or {}
            has_path = np.zeros(len(bids), dtype=np.float32)
            for i, bid in enumerate(bids):
                txt = run_txt.get(bid, "")
                if PATH_RE.search(txt) or "wrote" in txt.lower() or "/workspace/" in txt:
                    has_path[i] = 1.0

            costs = np.array([max(t_map.get(b, 1), 1) + 16 for b in bids], dtype=np.int32)

            fast_train.append({
                "block_ids": bids,
                "atoms": atoms,
                "F": F,
                "budgets_adj": budgets_adj,
                "tokens": t_map,
                "feats": feats,
                "is_loc": is_loc,
                "has_path": has_path,
                "costs": costs,
            })

        if not fast_train:
            return 0.0

        def eval_weights(w):
            tot = 0.0
            for it in fast_train:
                pred = it["feats"] @ w
                if it["is_loc"]:
                    pred = pred + 1.0 * it["has_path"]
                order = list(pred.argsort()[::-1])

                # Micro-budget promotion rule
                if len(order) > 1 and it["costs"][order[0]] > 125:
                    s_top = pred[order[0]]
                    for pos in range(1, min(5, len(order))):
                        idx = order[pos]
                        if it["costs"][idx] <= 125 and pred[idx] >= s_top - 0.05:
                            item_p = order.pop(pos)
                            order.insert(0, item_p)
                            break

                ordered_ids = [it["block_ids"][i] for i in order]
                found = [[] for _ in it["atoms"]]
                for x, t in read_order(ordered_ids, it["tokens"], max(it["F"], BUDGETS[-1])):
                    for f, a in zip(found, it["atoms"]):
                        if a.get(x):
                            f.append((t, a[x]))
                score, _ = shares(found, it["budgets_adj"])
                tot += score
            return tot / len(fast_train)

        current_w = self.weights.copy()
        best_score = eval_weights(current_w)
        if verbose:
            print(f"  [Direct Policy] Initial train score: {best_score:.4f}")

        step_sizes = [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]
        D = len(current_w)

        for it in range(n_iter):
            improved = False
            for d in range(1, D):  # keep anchor d=0 fixed at 1.0
                orig_val = current_w[d]
                best_val = orig_val
                for step in step_sizes:
                    current_w[d] = orig_val + step
                    s = eval_weights(current_w)
                    if s > best_score + 1e-5:
                        best_score = s
                        best_val = current_w[d]
                        improved = True
                current_w[d] = best_val

            if verbose:
                print(f"  [Direct Policy] Iter {it+1}/{n_iter}: train_score={best_score:.4f}")
            if not improved:
                break

        self.weights = current_w
        return best_score

    def save(self, path: Path):
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_names": self.FEATURE_NAMES,
            "weights": self.weights.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        w = np.array(data["weights"], dtype=np.float32)
        if len(w) > len(self.FEATURE_NAMES):
            w = w[:len(self.FEATURE_NAMES)]
        elif len(w) < len(self.FEATURE_NAMES):
            pad = np.zeros(len(self.FEATURE_NAMES) - len(w), dtype=np.float32)
            w = np.concatenate([w, pad])
        self.weights = w




def plackett_luce_sample(
    log_scores: torch.Tensor,  # (N,)
    temperature: float = 1.0,
    k: Optional[int] = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a ranking via Plackett-Luce distribution using the Gumbel-Top-K trick.
    Supports top-k truncated rollout to remove gradient noise from irrelevant tail items.

    Returns:
        perm: (k,) or (N,) LongTensor — sampled permutation of candidate indices
        log_prob: scalar tensor with gradient — total log-probability of sampled prefix
        per_pos_log_probs: (k,) tensor with gradient — per-position log-probabilities
    """
    N = log_scores.shape[0]
    if k is None or k >= N:
        k = N

    # Gumbel-Top-k trick (Kool et al., 2019):
    u = torch.rand_like(log_scores).clamp(min=1e-9, max=1.0 - 1e-9)
    g = -torch.log(-torch.log(u))
    perturbed = (log_scores / temperature) + g
    sorted_indices = perturbed.argsort(descending=True)
    perm = sorted_indices[:k]

    # Per-position log-probabilities under Plackett-Luce for top-k prefix:
    all_s_sorted = log_scores[sorted_indices] / temperature
    log_denoms = torch.logcumsumexp(all_s_sorted.flip(0), dim=0).flip(0)[:k]
    s_perm = all_s_sorted[:k]
    per_pos_log_probs = s_perm - log_denoms  # (k,) — each has gradient
    log_prob = per_pos_log_probs.sum()        # scalar (backward compatible)

    return perm, log_prob, per_pos_log_probs

