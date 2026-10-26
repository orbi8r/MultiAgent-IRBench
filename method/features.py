"""
features.py — Structural feature extraction (15-d per block/event).

As specified in Section 4.6 of the plan.
"""
import math
import re
from typing import Dict, List

import numpy as np

PATH_RE = re.compile(r"(?:/workspace/)?[\w-]+/[\w./-]+\.\w+")
CODE_RE = re.compile(r"\b(def|class|import|from|return|if|else|elif|for|while)\b|```")

FEATURE_NAMES = [
    "tokens_log",           # 0
    "text_len_log",         # 1
    "kind_is_model",        # 2
    "kind_is_tool_call",    # 3
    "kind_is_tool_result",  # 4
    "shift",                # 5
    "position_frac",        # 6
    "in_degree",            # 7
    "out_degree",           # 8
    "has_file_write",       # 9
    "has_file_read",        # 10
    "has_path",             # 11
    "has_code",             # 12
    "conv_agent_count",     # 13
    "is_copy_target",       # 14
]
FEATURE_DIM = len(FEATURE_NAMES)  # 15


def extract_block_features(
    block: dict,
    event_map: dict,
    graph: dict,
) -> np.ndarray:
    """
    Compute the 15-d structural feature vector for a single block.

    Args:
        block:      one block dict (id, event, tokens, text)
        event_map:  {event_id: event_dict}
        graph:      output of build_graph()

    Returns:
        float32 array of shape (15,)
    """
    eid = block["event"]
    event = event_map.get(eid, {})
    kind = event.get("kind", "")
    shift = float(event.get("shift", 0))

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 0: tokens_log
    feat[0] = math.log(1 + block.get("tokens", 0))
    # 1: text_len_log
    feat[1] = math.log(1 + len(block.get("text", "")))
    # 2-4: kind one-hot
    feat[2] = 1.0 if kind == "model" else 0.0
    feat[3] = 1.0 if kind == "tool_call" else 0.0
    feat[4] = 1.0 if kind == "tool_result" else 0.0
    # 5: shift
    feat[5] = shift
    # 6: position_frac
    feat[6] = graph.get("temporal_frac", {}).get(eid, 0.0)
    # 7: in_degree
    feat[7] = float(graph.get("in_degree", {}).get(eid, 0))
    # 8: out_degree
    feat[8] = float(graph.get("out_degree", {}).get(eid, 0))
    # 9: has_file_write
    feat[9] = 1.0 if eid in graph.get("file_write_sources", set()) else 0.0
    # 10: has_file_read
    feat[10] = 1.0 if eid in graph.get("file_read_targets", set()) else 0.0
    # 11: has_path
    feat[11] = 1.0 if PATH_RE.search(block.get("text", "")) else 0.0
    # 12: has_code
    feat[12] = 1.0 if CODE_RE.search(block.get("text", "")) else 0.0
    # 13: conv_agent_count
    conv = event.get("conv", "")
    feat[13] = float(len(graph.get("agents_by_conv", {}).get(conv, set())))
    # 14: is_copy_target
    feat[14] = 1.0 if eid in graph.get("copy_targets", set()) else 0.0

    return feat


def extract_all_block_features(
    blocks: list,
    events: list,
    graph: dict,
) -> Dict[str, np.ndarray]:
    """
    Compute features for all blocks in a run.

    Returns:
        {block_id: float32 array (15,)}
    """
    event_map = {e["id"]: e for e in events}
    return {b["id"]: extract_block_features(b, event_map, graph) for b in blocks}


def normalise_features(feat_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Z-score normalise features across all blocks (per feature dimension).
    Returns a new dict with normalised arrays.
    """
    if not feat_dict:
        return feat_dict
    mat = np.stack(list(feat_dict.values()), axis=0)  # (N, 15)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0) + 1e-8
    normed = {bid: (feat - mean) / std for bid, feat in feat_dict.items()}
    return normed
