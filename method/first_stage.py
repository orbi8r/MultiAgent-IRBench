"""
first_stage.py — Module 1: Structure-Aware First-Stage Retrieval.

Combines:
  1. Lexical BM25 over raw block texts
  2. Field-based BM25 over structured metadata (agent name, step kind, touched file)
  3. Dense retrieval (MiniLM cosine similarity) via Reciprocal Rank Fusion (RRF)
  4. Kind Multipliers (tool_call x2.0, tool_result x0.3, prompt x0.0, model x1.0)
  5. Move Following (tracking mv/cp paths for location questions)
  6. Neighbour Score Expansion (same-event sharing x0.5, agent step sharing x0.3)
  7. Exact Text Deduplication (normalised line SHA-1 hashing)

Memory-efficient:
  - Embeddings stay as CPU numpy float32
  - Shared SentenceTransformer loaded once
"""
import math
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from method.config import (
    BM25_B, BM25_K1, DENSE_BATCH_SIZE, DENSE_MODEL_NAME,
    RRF_K, STRUCT_WEIGHTS, TOP_K_CANDIDATES, CACHE_DIR
)
from method.features import FEATURE_NAMES
from method.graph_builder import (
    hash_block, parse_write_path, parse_read_path, parse_echo_path
)
from method.minhash import MinHashEngine

# Global shared SentenceTransformer model — loaded once, reused across all runs
_SHARED_MODEL = None


def get_shared_model():
    """Return (or create) the single shared MiniLM model on CPU."""
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [Dense] Loading MiniLM model ({DENSE_MODEL_NAME})...")
        _SHARED_MODEL = SentenceTransformer(DENSE_MODEL_NAME, device="cpu")
        _SHARED_MODEL.eval()
        print("  [Dense] Model loaded on CPU.")
    return _SHARED_MODEL


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


PATH_RE = re.compile(r"(?:/workspace/)?([a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+\.[a-zA-Z0-9_]+)")
MV_RE = re.compile(r"\bmv\s+([^\s;]+)\s+([^\s;]+)")
KIND_MULTIPLIERS = {
    "tool_call": 2.0,
    "tool_result": 0.3,
    "prompt": 0.0,
    "model": 1.0,
}


class BM25Index:
    """Okapi BM25 index over arbitrary text lists."""

    def __init__(self, texts: List[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self._build(texts)

    def _build(self, texts: List[str]):
        self.docs = [Counter(_tokenise(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(w for d in self.docs for w in d)
        self.n = len(self.docs)
        self.postings: Dict[str, List[int]] = defaultdict(list)
        for i, d in enumerate(self.docs):
            for w in d:
                self.postings[w].append(i)

    def score_dict(self, query: str) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        for w in set(_tokenise(query)):
            df_w = self.df.get(w, 0)
            if df_w == 0:
                continue
            idf = math.log(1 + (self.n - df_w + 0.5) / (df_w + 0.5))
            for i in self.postings.get(w, []):
                tf = self.docs[i][w]
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def score(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        scores = self.score_dict(query)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


def _gpu_encode(model, texts: list, batch_size: int = 512) -> "np.ndarray":
    """Encode texts using GPU (fast), return CPU numpy float32."""
    import torch
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    if use_gpu:
        model = model.to("cuda")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    if use_gpu:
        model.to("cpu")
        torch.cuda.empty_cache()
    return embeddings


class DenseIndex:
    """MiniLM-based dense retrieval index with CPU numpy storage and disk caching."""

    def __init__(self, blocks: list, batch_size: int = 512):
        self.blocks = blocks
        self._model_ref = get_shared_model()

        # Check disk cache: unique key based on first block ID and total block count
        cache_file = None
        if blocks:
            b0_id = str(blocks[0].get("id", "0")).replace("/", "_")
            cache_file = CACHE_DIR / f"dense_{b0_id}_{len(blocks)}.npy"
            if cache_file.exists():
                t_load = time.time()
                self.embeddings = np.load(cache_file)
                print(f"  [Dense] Loaded {len(blocks)} embeddings from disk cache {cache_file.name} in {time.time()-t_load:.2f}s.")
                return

        texts = [b["text"][:512] for b in blocks]
        import torch
        actual_bs = batch_size if torch.cuda.is_available() else 64
        print(f"  [Dense] Encoding {len(texts)} blocks (batch_size={actual_bs})...")
        self.embeddings = _gpu_encode(self._model_ref, texts, actual_bs)

        if cache_file:
            np.save(cache_file, self.embeddings)
            print(f"  [Dense] Cached {len(blocks)} embeddings to {cache_file.name}.")

    def encode_query(self, query: str) -> np.ndarray:
        return self._model_ref.encode(
            [query[:512]],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )[0]

    def score(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        q_emb = self.encode_query(query)
        sims = self.embeddings @ q_emb
        ranked = sorted(enumerate(sims.tolist()), key=lambda x: -x[1])
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def free(self):
        del self.embeddings
        self.embeddings = None
        import gc
        gc.collect()


class FirstStageRetriever:
    """
    High-Performance First-Stage Retriever implementing the complete
    heuristic stack:
      - Block BM25 + Field BM25 + Dense RRF fusion
      - Step kind multipliers (tool_call x2.0, tool_result x0.3, prompt x0.0)
      - Location move-following
      - Neighbour expansion (same-event x0.5 + agent step x0.3)
      - Exact text deduplication
    """

    def __init__(
        self,
        blocks: list,
        block_features: Optional[Dict[str, np.ndarray]] = None,
        events: Optional[list] = None,
        use_dense: bool = True,
        minhash_threshold: Optional[float] = 0.85,
    ):
        self.blocks = blocks
        self.block_ids = [b["id"] for b in blocks]
        self.block_features = block_features or {}
        self.use_dense = use_dense
        self.events = events or []
        self.minhash_threshold = minhash_threshold

        # 1. Block BM25
        self.bm25 = BM25Index([b["text"] for b in blocks])

        # 2. Dense Index
        if use_dense:
            self.dense = DenseIndex(blocks)
        else:
            self.dense = None

        # 3. Field BM25 & Structural mappings if events provided
        self.event_map = {e["id"]: e for e in self.events}
        self.block_event = {
            b["id"]: b.get("event") or b["id"].split("/")[0] for b in blocks
        }

        # Build metadata field texts for field-based BM25
        field_texts = []
        for b in blocks:
            e = self.event_map.get(self.block_event[b["id"]], {})
            ag = e.get("agent", "")
            kd = e.get("kind", "")
            ft = parse_write_path(e) or parse_read_path(e) or parse_echo_path(e) or ""
            if not ft:
                m = PATH_RE.search(e.get("text", ""))
                if m:
                    ft = m.group(1)
            field_texts.append(f"{ag} {kd} {ft}")
        self.field_bm25 = BM25Index(field_texts)

        # Precompute block hashes for deduplication
        self.block_hashes = [hash_block(b["text"]) for b in blocks]

        # MinHash Engine for near-duplicate deduplication
        self.minhash_engine = MinHashEngine(num_perm=128, threshold=minhash_threshold or 0.85)
        cache_key = f"minhash_{self.block_ids[0].replace('/', '_')}_{len(blocks)}.npy" if self.block_ids else None
        cache_path = CACHE_DIR / cache_key if cache_key else None
        if cache_path and cache_path.exists():
            self.minhash_signatures = np.load(cache_path)
        else:
            self.minhash_signatures = self.minhash_engine.compute_all(blocks)
            if cache_path:
                try:
                    np.save(cache_path, self.minhash_signatures)
                except Exception:
                    pass

        # Group blocks by event
        self.event_blocks = defaultdict(list)
        for i, b in enumerate(blocks):
            self.event_blocks[self.block_event[b["id"]]].append(i)

        # Agent event sequence for temporal neighbor expansion
        events_by_ts = sorted(self.events, key=lambda e: e.get("ts", 0))
        self.agent_events = defaultdict(list)
        for e in events_by_ts:
            self.agent_events[e.get("agent")].append(e["id"])

        self.event_to_agent_pos = {
            eid: (ag, pos)
            for ag, eids in self.agent_events.items()
            for pos, eid in enumerate(eids)
        }

        # Move following mapping (mv src dst)
        self.moves_map = {}
        for e in events_by_ts:
            if e.get("kind") == "tool_call":
                for m in MV_RE.finditer(e.get("text", "")):
                    src = m.group(1).rstrip("/")
                    dst = m.group(2).rstrip("/")
                    src_base = src.split("/")[-1]
                    if "/" not in dst or dst.endswith(".md") or dst.endswith(".log") or dst.endswith(".txt"):
                        dst_full = dst
                    else:
                        dst_full = f"{dst}/{src_base}"
                    self.moves_map[src] = dst_full
                    self.moves_map[src_base] = dst_full

        # Lookup mapping: block_id -> index
        self._id_to_idx = {bid: i for i, bid in enumerate(self.block_ids)}

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K_CANDIDATES,
        qtype: str = "",
        minhash_threshold: Optional[float] = -1.0,
    ) -> List[str]:
        """Retrieve top_k block IDs applying the full heuristic pipeline."""
        qtext = question

        # 1. BM25 rank
        b_scores = self.bm25.score_dict(qtext)
        b_ranked = sorted(b_scores, key=lambda i: -b_scores[i])
        b_rank = {idx: r for r, idx in enumerate(b_ranked)}

        # 2. Field BM25 rank
        f_scores = self.field_bm25.score_dict(qtext)
        f_ranked = sorted(f_scores, key=lambda i: -f_scores[i])
        f_rank = {idx: r for r, idx in enumerate(f_ranked)}

        # 3. Dense cosine rank
        d_rank = {}
        if self.dense is not None:
            q_emb = self.dense.encode_query(qtext)
            sims = self.dense.embeddings @ q_emb
            d_ranked = np.argsort(-sims).tolist()
            d_rank = {idx: r for r, idx in enumerate(d_ranked)}

        # Reciprocal Rank Fusion (RRF)
        pool = set(b_ranked[:600]) | set(f_ranked[:600])
        if self.dense is not None:
            pool |= set(d_ranked[:600])

        rrf = defaultdict(float)
        for idx in pool:
            if idx in b_rank:
                rrf[idx] += 1.0 / (RRF_K + b_rank[idx])
            if idx in f_rank:
                rrf[idx] += 0.5 / (RRF_K + f_rank[idx])
            if idx in d_rank:
                rrf[idx] += 1.0 / (RRF_K + d_rank[idx])

        # 4. Kind Multipliers
        comb = {}
        for idx, sc in rrf.items():
            eid = self.block_event[self.blocks[idx]["id"]]
            k = self.event_map.get(eid, {}).get("kind", "model")
            comb[idx] = sc * KIND_MULTIPLIERS.get(k, 1.0)

        # 5. Move Following for location queries
        is_loc = qtype == "location" or qtext.lower().startswith("where") or "location" in qtext.lower()
        if is_loc:
            for src, dst in self.moves_map.items():
                if src in qtext or src.lower() in qtext.lower():
                    dst_base = dst.split("/")[-1]
                    for idx in range(len(self.blocks)):
                        btext = self.blocks[idx]["text"]
                        if dst in btext or dst_base in btext:
                            comb[idx] = max(comb.get(idx, 0.0), 0.02)

        # 6. Neighbour Score Expansion
        # Same-event sharing: each block gets max(0.5 * event_max, own_score)
        ev_max = defaultdict(float)
        for idx, sc in comb.items():
            ev = self.block_event[self.blocks[idx]["id"]]
            if sc > ev_max[ev]:
                ev_max[ev] = sc

        for ev, mx in ev_max.items():
            sh_same = mx * 0.5
            for idx in self.event_blocks[ev]:
                if sh_same > comb.get(idx, 0.0):
                    comb[idx] = sh_same

            # Agent consecutive step sharing: 0.3 * adjacent_event_max
            if ev in self.event_to_agent_pos:
                ag, pos = self.event_to_agent_pos[ev]
                eids = self.agent_events[ag]
                sh_adj = mx * 0.3
                for delta in [-1, 1]:
                    if 0 <= pos + delta < len(eids):
                        adj_ev = eids[pos + delta]
                        for idx in self.event_blocks[adj_ev]:
                            if sh_adj > comb.get(idx, 0.0):
                                comb[idx] = sh_adj

        # 7. Deduplication (MinHash near-duplicate or exact content hash suppression)
        best = sorted(comb, key=lambda idx: -comb[idx])
        seen_hashes = set()
        deduped_ids = []
        th = self.minhash_threshold if minhash_threshold == -1.0 else minhash_threshold
        if th is not None and self.minhash_signatures is not None:
            threshold_matches = int(round(th * self.minhash_engine.num_perm))
            selected_sigs = np.empty((top_k, self.minhash_engine.num_perm), dtype=np.int64)
            for idx in best:
                h = self.block_hashes[idx]
                if h in seen_hashes:
                    continue
                cand_sig = self.minhash_signatures[idx]
                n_sel = len(deduped_ids)
                if n_sel > 0:
                    matches = (selected_sigs[:n_sel] == cand_sig).sum(axis=1)
                    if np.any(matches >= threshold_matches):
                        continue
                seen_hashes.add(h)
                selected_sigs[n_sel] = cand_sig
                deduped_ids.append(self.block_ids[idx])
                if len(deduped_ids) >= top_k:
                    break
        else:
            for idx in best:
                h = self.block_hashes[idx]
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    deduped_ids.append(self.block_ids[idx])
                    if len(deduped_ids) >= top_k:
                        break

        return deduped_ids

    def retrieve_with_embeddings(
        self,
        question: str,
        top_k: int = TOP_K_CANDIDATES,
        qtype: str = "",
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Returns (block_ids, query_emb, block_embs) — CPU numpy arrays.
        """
        if self.dense is None:
            raise RuntimeError("Dense index required for embedding retrieval.")

        block_ids = self.retrieve(question, top_k=top_k, qtype=qtype)
        q_emb = self.dense.encode_query(question)

        indices = [self._id_to_idx[bid] for bid in block_ids if bid in self._id_to_idx]
        block_embs = (
            self.dense.embeddings[indices]
            if indices
            else np.zeros((0, 384), dtype=np.float32)
        )

        return block_ids, q_emb, block_embs

    def retrieve_with_scores(
        self,
        question: str,
        top_k: int = TOP_K_CANDIDATES,
        qtype: str = "",
        minhash_threshold: Optional[float] = -1.0,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (block_ids, query_emb, block_embs, retrieval_feats)
        where retrieval_feats is (N, 5) float32:
          [0] bm25_score_norm     — min-max normalized block BM25 score
          [1] field_bm25_norm     — min-max normalized field BM25 score
          [2] dense_cosine_sim    — raw cosine similarity with query
          [3] rrf_score_norm      — min-max normalized RRF fusion score
          [4] first_stage_rank_inv — inverse rank: 1/(1+rank) in (0, 1]
        """
        if self.dense is None:
            raise RuntimeError("Dense index required for scored retrieval.")

        qtext = question

        # 1. Block BM25
        b_scores = self.bm25.score_dict(qtext)

        # 2. Field BM25
        f_scores = self.field_bm25.score_dict(qtext)

        # 3. Dense cosine
        q_emb = self.dense.encode_query(qtext)
        sims = self.dense.embeddings @ q_emb  # (M,)

        # 4. RRF fusion
        b_ranked = sorted(b_scores, key=lambda i: -b_scores[i])
        b_rank = {idx: r for r, idx in enumerate(b_ranked)}
        f_ranked = sorted(f_scores, key=lambda i: -f_scores[i])
        f_rank = {idx: r for r, idx in enumerate(f_ranked)}
        d_ranked = np.argsort(-sims).tolist()
        d_rank = {idx: r for r, idx in enumerate(d_ranked)}

        pool = set(b_ranked[:600]) | set(f_ranked[:600]) | set(d_ranked[:600])
        rrf = defaultdict(float)
        for idx in pool:
            if idx in b_rank:
                rrf[idx] += 1.0 / (RRF_K + b_rank[idx])
            if idx in f_rank:
                rrf[idx] += 0.5 / (RRF_K + f_rank[idx])
            if idx in d_rank:
                rrf[idx] += 1.0 / (RRF_K + d_rank[idx])

        # Kind multipliers
        comb = {}
        for idx, sc in rrf.items():
            eid = self.block_event[self.blocks[idx]["id"]]
            k = self.event_map.get(eid, {}).get("kind", "model")
            comb[idx] = sc * KIND_MULTIPLIERS.get(k, 1.0)

        # Move following
        is_loc = qtype == "location" or qtext.lower().startswith("where") or "location" in qtext.lower()
        if is_loc:
            for src, dst in self.moves_map.items():
                if src in qtext or src.lower() in qtext.lower():
                    dst_base = dst.split("/")[-1]
                    for idx in range(len(self.blocks)):
                        btext = self.blocks[idx]["text"]
                        if dst in btext or dst_base in btext:
                            comb[idx] = max(comb.get(idx, 0.0), 0.02)

        # Neighbour expansion
        ev_max = defaultdict(float)
        for idx, sc in comb.items():
            ev = self.block_event[self.blocks[idx]["id"]]
            if sc > ev_max[ev]:
                ev_max[ev] = sc
        for ev, mx in ev_max.items():
            sh_same = mx * 0.5
            for idx in self.event_blocks[ev]:
                if sh_same > comb.get(idx, 0.0):
                    comb[idx] = sh_same
            if ev in self.event_to_agent_pos:
                ag, pos = self.event_to_agent_pos[ev]
                eids = self.agent_events[ag]
                sh_adj = mx * 0.3
                for delta in [-1, 1]:
                    if 0 <= pos + delta < len(eids):
                        adj_ev = eids[pos + delta]
                        for idx in self.event_blocks[adj_ev]:
                            if sh_adj > comb.get(idx, 0.0):
                                comb[idx] = sh_adj

        # Deduplication (MinHash near-duplicate or exact content hash suppression)
        best = sorted(comb, key=lambda idx: -comb[idx])
        seen_hashes = set()
        selected_indices = []
        th = self.minhash_threshold if minhash_threshold == -1.0 else minhash_threshold
        if th is not None and self.minhash_signatures is not None:
            threshold_matches = int(round(th * self.minhash_engine.num_perm))
            selected_sigs = np.empty((top_k, self.minhash_engine.num_perm), dtype=np.int64)
            for idx in best:
                h = self.block_hashes[idx]
                if h in seen_hashes:
                    continue
                cand_sig = self.minhash_signatures[idx]
                n_sel = len(selected_indices)
                if n_sel > 0:
                    matches = (selected_sigs[:n_sel] == cand_sig).sum(axis=1)
                    if np.any(matches >= threshold_matches):
                        continue
                seen_hashes.add(h)
                selected_sigs[n_sel] = cand_sig
                selected_indices.append(idx)
                if len(selected_indices) >= top_k:
                    break
        else:
            for idx in best:
                h = self.block_hashes[idx]
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    selected_indices.append(idx)
                    if len(selected_indices) >= top_k:
                        break

        block_ids = [self.block_ids[idx] for idx in selected_indices]
        N = len(selected_indices)

        # Build block embeddings
        block_embs = (
            self.dense.embeddings[selected_indices]
            if selected_indices
            else np.zeros((0, 384), dtype=np.float32)
        )

        # Build retrieval feature vectors (N, 10)
        retrieval_feats = np.zeros((N, 10), dtype=np.float32)
        if N > 0:
            # Gather raw scores for selected candidates
            raw_bm25 = np.array([b_scores.get(idx, 0.0) for idx in selected_indices], dtype=np.float32)
            raw_field = np.array([f_scores.get(idx, 0.0) for idx in selected_indices], dtype=np.float32)
            raw_dense = np.array([sims[idx] for idx in selected_indices], dtype=np.float32)
            raw_rrf = np.array([comb.get(idx, 0.0) for idx in selected_indices], dtype=np.float32)

            # Min-max normalize to [0, 1]
            def _norm(arr):
                mn, mx = arr.min(), arr.max()
                if mx - mn > 1e-9:
                    return (arr - mn) / (mx - mn)
                return np.zeros_like(arr)

            retrieval_feats[:, 0] = _norm(raw_bm25)
            retrieval_feats[:, 1] = _norm(raw_field)
            retrieval_feats[:, 2] = raw_dense  # cosine sim is already in [-1, 1]
            retrieval_feats[:, 3] = _norm(raw_rrf)
            retrieval_feats[:, 4] = 1.0 / (1.0 + np.arange(N, dtype=np.float32))  # inverse rank
            # 5. Log-discount rank (DCG prior)
            retrieval_feats[:, 5] = 1.0 / np.log2(2.0 + np.arange(N, dtype=np.float32))
            # 6. Normalized token cost
            block_tokens = np.array([max(self.blocks[idx].get("tokens", 64), 64) for idx in selected_indices], dtype=np.float32)
            retrieval_feats[:, 6] = np.clip(block_tokens / 1500.0, 0.0, 1.0)
            # 7. Token cost efficiency: (dense + rrf) / log2(1 + tokens / 64)
            cost_factor = np.log2(1.0 + block_tokens / 64.0)
            retrieval_feats[:, 7] = (retrieval_feats[:, 2] + retrieval_feats[:, 3]) / np.maximum(cost_factor, 1.0)
            # 8. Exact path match in question
            import re
            path_tokens = [tok.lower() for tok in re.findall(r'[\w\-\./]+\.[a-zA-Z0-9]+', qtext) if len(tok) > 3]
            if path_tokens:
                path_matches = [1.0 if any(pt in self.blocks[idx]["text"].lower() for pt in path_tokens) else 0.0 for idx in selected_indices]
                retrieval_feats[:, 8] = np.array(path_matches, dtype=np.float32)
            # 9. Kind is tool_call (holds 62% of answer credit)
            kind_is_tc = [1.0 if self.event_map.get(self.block_event[self.blocks[idx]["id"]], {}).get("kind", "") == "tool_call" else 0.0 for idx in selected_indices]
            retrieval_feats[:, 9] = np.array(kind_is_tc, dtype=np.float32)

        return block_ids, q_emb, block_embs, retrieval_feats

    def free(self):
        """Release dense embedding memory."""
        if self.dense is not None:
            self.dense.free()

