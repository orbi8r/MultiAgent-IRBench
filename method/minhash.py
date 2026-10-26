"""
minhash.py — MinHash Near-Duplicate Deduplication Engine for MultiAgent-IRBench.

Implements MinHash LSH and signature-based Jaccard similarity estimation for
agent traces, tool outputs, terminal logs, and code blocks.
Zero external C dependencies (uses standard zlib.crc32 and numpy).
"""
import re
import zlib
from typing import List, Optional, Set
import numpy as np


class MinHashEngine:
    """
    MinHash signature generator and near-duplicate detector.
    Uses M hash permutations over word unigrams + bigrams (and character 4-grams for compact code).
    """
    PRIME = 2147483647  # 2^31 - 1 (Mersenne prime)

    def __init__(self, num_perm: int = 128, threshold: float = 0.85, seed: int = 42):
        self.num_perm = num_perm
        self.threshold = threshold
        self.threshold_matches = int(round(threshold * num_perm))

        rng = np.random.RandomState(seed)
        self.A = rng.randint(1, self.PRIME, size=(1, num_perm), dtype=np.int64)
        self.B = rng.randint(0, self.PRIME, size=(1, num_perm), dtype=np.int64)

    def compute_signature(self, text: str) -> np.ndarray:
        """Compute an (M,) int64 MinHash signature for a text block."""
        words = text.strip().lower().split()
        if not words:
            return np.zeros(self.num_perm, dtype=np.int64)

        shingles = set(words)
        # Word bigrams
        for i in range(len(words) - 1):
            shingles.add(f"{words[i]} {words[i+1]}")

        # If very few words but substantial character length (e.g. minified code or long token),
        # add character 4-grams to avoid degenerate single-shingle sets.
        if len(words) < 4 and len(text) > 16:
            clean = re.sub(r"\s+", "", text.lower())
            for i in range(len(clean) - 3):
                shingles.add(clean[i:i+4])

        hashes = np.array([zlib.crc32(s.encode("utf-8")) for s in shingles], dtype=np.int64).reshape(-1, 1)
        # Vectorized linear hash permutation: (hashes * A + B) % PRIME
        permuted = (hashes * self.A + self.B) % self.PRIME
        return permuted.min(axis=0)

    def compute_all(self, blocks: list) -> np.ndarray:
        """
        Compute MinHash signatures for an entire run of blocks.
        Returns: (N, M) int64 matrix of signatures.
        """
        N = len(blocks)
        sigs = np.empty((N, self.num_perm), dtype=np.int64)
        for i, b in enumerate(blocks):
            sigs[i] = self.compute_signature(b["text"])
        return sigs

    def is_near_duplicate(
        self,
        cand_sig: np.ndarray,
        selected_sigs: np.ndarray,
        n_selected: int,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if cand_sig has Jaccard similarity >= threshold with any vector in selected_sigs[:n_selected].
        """
        if n_selected == 0:
            return False
        req_matches = self.threshold_matches if threshold is None else int(round(threshold * self.num_perm))
        matches = (selected_sigs[:n_selected] == cand_sig).sum(axis=1)
        return bool(np.any(matches >= req_matches))
