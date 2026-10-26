"""
budget_selector.py — Module 4: Calibrated Budget-Aware Sequential Selection.

Converts reranked neural scores into a final submission list that maximises
coverage under the 1k/2k/4k/8k token budget constraints.

Calibrated formulation:
  1. Sigmoid utility mapping: sigma(s) = 1 / (1 + exp(-s)) > 0
  2. Sub-linear cost damping: Utility = sigma(s) / (cost ** alpha)
     where alpha = BUDGET_COST_ALPHA (default 0.01) prevents length over-penalization.
  3. Optional same-event diversity penalty.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from method.config import (
    BUDGET_COST_ALPHA, BUDGET_USE_DIVERSITY,
    DIVERSITY_SAME_EVENT_PENALTY, MIN_COST
)


def select_budget_aware(
    candidate_ids: List[str],
    scores: List[float],
    tokens_map: Dict[str, int],
    use_diversity_penalty: bool = BUDGET_USE_DIVERSITY,
    cost_alpha: float = BUDGET_COST_ALPHA,
    max_items: int = 300,
) -> List[str]:
    """
    Apply calibrated value-density ordering with damped cost exponent.

    Formula:
        Utility = sigma(score) * diversity_penalty(event) / (cost ** alpha)
    where:
        sigma(s) = 1 / (1 + exp(-s)) maps unbounded logits to positive utilities
        alpha = calibrated damping exponent (default 0.01)

    Returns:
        ordered list of IDs, best first
    """
    if not candidate_ids:
        return []

    # Map raw logits to positive probabilities in (0, 1)
    probs = 1.0 / (1.0 + np.exp(-np.array(scores, dtype=np.float32)))
    costs = np.array([max(tokens_map.get(cid, MIN_COST), MIN_COST) for cid in candidate_ids], dtype=np.float32)

    # Fast vectorized sort when diversity decay is disabled (recommended)
    if not use_diversity_penalty or DIVERSITY_SAME_EVENT_PENALTY == [1.0, 1.0, 1.0]:
        values = probs / (costs ** cost_alpha)
        idx = np.argsort(-values)
        return [candidate_ids[i] for i in idx[:max_items]]

    # Greedy selection with event-level diversity decay
    def event_of(cid: str) -> str:
        return cid.split("/")[0]

    event_block_count: Dict[str, int] = defaultdict(int)
    remaining = list(zip(candidate_ids, probs, costs))
    final_order = []

    for _ in range(min(max_items, len(candidate_ids))):
        if not remaining:
            break
        best_val = -1e9
        best_idx = 0
        for i, (cid, p, c) in enumerate(remaining):
            ev = event_of(cid)
            n = event_block_count[ev]
            pen = DIVERSITY_SAME_EVENT_PENALTY[min(n, len(DIVERSITY_SAME_EVENT_PENALTY) - 1)]
            val = (p * pen) / (c ** cost_alpha)
            if val > best_val:
                best_val = val
                best_idx = i

        cid, _, _ = remaining.pop(best_idx)
        final_order.append(cid)
        event_block_count[event_of(cid)] += 1

    return final_order


def build_submission(
    questions: list,
    retrieval_results: Dict[str, Tuple[List[str], List[float]]],
    tokens_map: Dict[str, int],
    use_diversity: bool = BUDGET_USE_DIVERSITY,
    cost_alpha: float = BUDGET_COST_ALPHA,
) -> Dict[str, List[str]]:
    """
    Build submission dict {qid: [ranked_ids...]} for all questions in run.
    """
    submission = {}
    for q in questions:
        qid = q["qid"]
        cand_ids, scores = retrieval_results.get(qid, ([], []))
        if not cand_ids:
            submission[qid] = []
            continue
        ordered = select_budget_aware(
            cand_ids, scores, tokens_map,
            use_diversity_penalty=use_diversity,
            cost_alpha=cost_alpha,
        )
        submission[qid] = ordered
    return submission
