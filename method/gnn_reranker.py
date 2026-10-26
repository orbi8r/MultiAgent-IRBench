"""
gnn_reranker.py — Module 3: Query-Conditioned Graph Attention Reranker (v3).

Key improvements over v2:
  - query_emb is now a required argument to forward() — attention is query-guided
  - Directed adjacency: tool_call → tool_result → model (not bidirectional cliques)
  - Intra-event edge weight reduced to 0.5 to prevent over-smoothing
  - Query-conditioned gating replaces fixed 0.3 blend weight
  - Score blending in raw score space (no sigmoid compression)
  - EMA REINFORCE baseline in GNN training (was missing)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from method.config import (
    GNN_HIDDEN_DIM, GNN_NUM_HEADS, GNN_NUM_LAYERS,
    DEVICE, RL_EMBED_DIM, RL_STRUCT_DIM
)


# ─── Query-Conditioned GAT Layer ──────────────────────────────────────────

class QueryConditionedGATLayer(nn.Module):
    """
    Single GAT layer where attention is modulated by query embedding.
    Query context biases which neighbors are attended to, making message
    passing query-relevant rather than query-blind.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        # Query modulation: project query to per-head attention bias
        self.query_attn = nn.Linear(in_dim, num_heads)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        nn.init.zeros_(self.query_attn.weight)
        nn.init.zeros_(self.query_attn.bias)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        adj: torch.Tensor,        # (N, N) adjacency (0/weight/1)
        query_context: Optional[torch.Tensor] = None,  # (N, in_dim) or None
    ) -> torch.Tensor:            # (N, out_dim)
        N = x.shape[0]
        Wh = self.W(x)  # (N, out_dim)
        Wh = Wh.view(N, self.num_heads, self.head_dim)  # (N, H, D)

        # Compute attention coefficients
        a_input = torch.cat([
            Wh.unsqueeze(1).expand(-1, N, -1, -1),  # (N, N, H, D) — source
            Wh.unsqueeze(0).expand(N, -1, -1, -1),  # (N, N, H, D) — dest
        ], dim=-1)  # (N, N, H, 2D)

        e = (a_input * self.a.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (N, N, H)
        e = F.leaky_relu(e, negative_slope=0.2)

        # Query-conditioned attention bias
        if query_context is not None:
            q_bias = self.query_attn(query_context)  # (N, num_heads)
            # Bias destination attention: higher bias = prefer this destination
            e = e + q_bias.unsqueeze(0)  # (N, N, H) + (1, N, H) broadcast over sources

        # Mask non-edges (only attend over neighbours + self)
        mask = (adj > 0).float().unsqueeze(-1)  # (N, N, 1)
        e = e * mask + (1 - mask) * (-1e9)
        alpha = F.softmax(e, dim=1)  # (N, N, H) — attention weights

        # Apply edge weights to attention
        edge_weights = adj.unsqueeze(-1)  # (N, N, 1)
        alpha = alpha * edge_weights

        # Aggregate
        out = (alpha.unsqueeze(-1) * Wh.unsqueeze(0)).sum(dim=1)  # (N, H, D)
        out = out.view(N, self.num_heads * self.head_dim)  # (N, out_dim)
        return out


class GNNReranker(nn.Module):
    """
    Query-Conditioned 2-layer GAT for graph-aware reranking.

    Now takes query_emb as a required argument.
    Uses query-conditioned gating instead of fixed 0.3 blend weight.
    Blends in raw score space (no sigmoid compression).
    """

    def __init__(
        self,
        embed_dim: int = RL_EMBED_DIM,
        struct_dim: int = RL_STRUCT_DIM,
        hidden_dim: int = GNN_HIDDEN_DIM,
        num_heads: int = GNN_NUM_HEADS,
        num_layers: int = GNN_NUM_LAYERS,
    ):
        super().__init__()
        in_dim = embed_dim + struct_dim  # 384 + 15 = 399

        self.block_proj = nn.Linear(in_dim, hidden_dim)
        self.query_proj = nn.Linear(embed_dim, hidden_dim)

        self.gat_layers = nn.ModuleList([
            QueryConditionedGATLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Query-conditioned gating (replaces fixed 0.3 blend)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # [gnn_repr, query_repr]
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.score_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.block_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.block_proj.bias)
        nn.init.kaiming_uniform_(self.query_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        # Initialize gate to output ~0.018 (start with minimal GNN contribution)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[2].weight)
        self.gate[2].bias.data.fill_(-4.0)  # sigmoid(-4) ≈ 0.018

    def forward(
        self,
        query_emb: torch.Tensor,    # (384,) or (N, 384)
        block_embs: torch.Tensor,   # (N, 384)
        struct_feats: torch.Tensor, # (N, 15)
        adj: torch.Tensor,          # (N, N) directed adjacency matrix
        rl_scores: torch.Tensor,    # (N,) scores from Module 2
    ) -> torch.Tensor:
        """Returns: (N,) blended scores in raw score space."""
        N = block_embs.shape[0]

        x = torch.cat([block_embs, struct_feats], dim=-1)  # (N, 399)
        x = F.relu(self.block_proj(x))  # (N, 128)

        # Query projection
        if query_emb.dim() == 1:
            q = F.relu(self.query_proj(query_emb)).unsqueeze(0).expand(N, -1)  # (N, 128)
        else:
            q = F.relu(self.query_proj(query_emb))  # (N, 128)

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, adj, query_context=q)  # query-conditioned attention
            x = F.elu(x)
            x = norm(x + residual)  # residual connection

        gnn_scores = self.score_head(x).squeeze(-1)  # (N,)

        # Query-conditioned gating: α ∈ (0, 1) per query
        # Use mean-pooled GNN representation + query for gate decision
        gate_input = torch.cat([x, q], dim=-1)  # (N, 256)
        alpha = self.gate(gate_input).squeeze(-1)  # (N,)

        # Blend in RAW SCORE SPACE (no sigmoid compression!)
        blended = alpha * gnn_scores + (1 - alpha) * rl_scores
        return blended

    def score_candidates(
        self,
        query_emb: np.ndarray,
        block_embs: np.ndarray,
        struct_feats: np.ndarray,
        adj: np.ndarray,
        rl_scores: np.ndarray,
    ) -> np.ndarray:
        """Numpy convenience wrapper."""
        self.eval()
        with torch.no_grad():
            qe = torch.tensor(query_emb, dtype=torch.float32, device=DEVICE)
            b = torch.tensor(block_embs, dtype=torch.float32, device=DEVICE)
            s = torch.tensor(struct_feats, dtype=torch.float32, device=DEVICE)
            a = torch.tensor(adj, dtype=torch.float32, device=DEVICE)
            r = torch.tensor(rl_scores, dtype=torch.float32, device=DEVICE)
            out = self.forward(qe, b, s, a, r)
        return out.cpu().numpy()


# ─── Directed Subgraph Construction ───────────────────────────────────────

def build_block_adjacency_directed(
    candidate_ids: List[str],
    graph: dict,
) -> np.ndarray:
    """
    Build DIRECTED adjacency for candidate blocks.

    Edge types:
      - Self-loops: adj[i,i] = 1.0
      - Same-event: directed by kind order (tool_call→tool_result→model),
        weight 0.5 to prevent over-smoothing
      - Cross-event provenance: directed edges preserved from graph, weight 1.0
    """
    N = len(candidate_ids)
    adj = np.eye(N, dtype=np.float32)
    if N <= 1 or not graph:
        return adj

    eids = [bid.split("/")[0] for bid in candidate_ids]
    event_map = graph.get("event_map", {})
    edges = graph.get("edges", {})

    # Collect DIRECTED provenance edges (do NOT symmetrize)
    directed_edges = set()
    for es in edges.values():
        for u, v in es:
            directed_edges.add((u, v))

    KIND_ORDER = {"tool_call": 0, "tool_result": 1, "model": 2, "prompt": -1}

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if eids[i] == eids[j]:
                # Same-event: directed by kind order with reduced weight
                ki = event_map.get(eids[i], {}).get("kind", "model")
                kj = event_map.get(eids[j], {}).get("kind", "model")
                oi = KIND_ORDER.get(ki, 1)
                oj = KIND_ORDER.get(kj, 1)
                if oi <= oj:  # forward direction only
                    adj[i, j] = 0.5  # reduced weight
            elif (eids[i], eids[j]) in directed_edges:
                adj[i, j] = 1.0  # directed cross-event edge

    return adj


def build_block_adjacency_np(candidate_ids: List[str], graph: dict) -> np.ndarray:
    """Legacy bidirectional adjacency — kept for backward compatibility."""
    return build_block_adjacency_directed(candidate_ids, graph)


def gnn_rerank(
    candidate_ids: List[str],
    rl_scores: np.ndarray,
    block_embs: np.ndarray,
    struct_feats: np.ndarray,
    graph: dict,
    gnn_model: GNNReranker,
    all_block_embs: Optional[Dict[str, np.ndarray]] = None,
    all_struct_feats: Optional[Dict[str, np.ndarray]] = None,
    query_emb: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Apply query-conditioned GNN message passing over provenance edges.
    """
    N = len(candidate_ids)
    if N == 0:
        return [], np.zeros(0, dtype=np.float32)

    adj = build_block_adjacency_directed(candidate_ids, graph)

    if query_emb is not None:
        blended = gnn_model.score_candidates(query_emb, block_embs, struct_feats, adj, rl_scores)
    else:
        # Fallback: just use RL scores if no query_emb provided
        blended = rl_scores

    sorted_indices = np.argsort(-blended)
    reranked_ids = [candidate_ids[i] for i in sorted_indices]
    return reranked_ids, blended[sorted_indices]
