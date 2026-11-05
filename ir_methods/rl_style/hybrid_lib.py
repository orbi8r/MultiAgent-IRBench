"""The original coordinate ascent (SubmodularDirectPolicy.fit in the original reranker module) and rank() generalised to any feature matrix, so an extra
column (the ftpub score) can sit beside the original 12 features. With the 12 columns it is the original code path: check_coordinate_ascent.py checks that
the weights are identical to SubmodularDirectPolicy.fit on the same items."""
import re
import numpy as np
from score import BUDGETS, ideal, read_order, shares
from method.reranker import extract_reranking_features

PATH_RE = re.compile(r"(?:/workspace/)?[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+\.[a-zA-Z0-9_]+")
STEPS = [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]


def is_loc(q, qtype):
    q = q.lower()
    return qtype == "location" or q.startswith("where") or "where is" in q or "location" in q


def her12(block_ids, rf, sf, tokens):
    f = extract_reranking_features(block_ids, rf, sf, tokens)
    return f[:, :12]


def path_flags(block_ids, run_text):
    h = np.zeros(len(block_ids), dtype=np.float32)
    for i, b in enumerate(block_ids):
        t = run_text.get(b, "")
        if PATH_RE.search(t) or "wrote" in t.lower() or "/workspace/" in t:
            h[i] = 1.0
    return h


def promote(order, pred, costs):
    """The original micro budget promotion rule (rank())."""
    order = list(order)
    if len(order) > 1 and costs[order[0]] > 125:
        s_top = pred[order[0]]
        for pos in range(1, min(5, len(order))):
            idx = order[pos]
            if costs[idx] <= 125 and pred[idx] >= s_top - 0.05:
                order.insert(0, order.pop(pos))
                break
    return order


def prep(item, feats):
    """item: block_ids, question, qtype, tokens_map, credits, run_text. feats (N, D) already built."""
    atoms = item["credits"]
    if not atoms:
        return None
    t = item["tokens_map"]
    Fc = ideal(atoms, t)[0]
    bids = item["block_ids"]
    return dict(block_ids=bids, atoms=atoms, F=Fc, budgets_adj=[max(Fc, b) for b in BUDGETS], tokens=t, feats=feats,
                is_loc=is_loc(item.get("question", ""), item.get("qtype", "")), has_path=path_flags(bids, item.get("run_text") or {}),
                costs=np.array([max(t.get(b, 1), 1) + 16 for b in bids], dtype=np.int32))


def eval_w(fast, w):
    tot = 0.0
    for it in fast:
        pred = it["feats"] @ w
        if it["is_loc"]:
            pred = pred + 1.0 * it["has_path"]
        order = promote(list(pred.argsort()[::-1]), pred, it["costs"])
        ids = [it["block_ids"][i] for i in order]
        found = [[] for _ in it["atoms"]]
        for x, t in read_order(ids, it["tokens"], max(it["F"], BUDGETS[-1])):
            for f, a in zip(found, it["atoms"]):
                if a.get(x):
                    f.append((t, a[x]))
        score, _ = shares(found, it["budgets_adj"])
        tot += score
    return tot / len(fast)


def fit_cd(fast, w0, n_iter=5):
    """Coordinate ascent exactly as the original fit(): anchor (column 0) fixed, steps and acceptance rule unchanged."""
    fast = [f for f in fast if f]
    w = np.array(w0, dtype=np.float32).copy()
    best = eval_w(fast, w)
    for _ in range(n_iter):
        improved = False
        for d in range(1, len(w)):
            orig, bv = w[d], w[d]
            for st in STEPS:
                w[d] = orig + st
                s = eval_w(fast, w)
                if s > best + 1e-5:
                    best, bv, improved = s, w[d], True
            w[d] = bv
        if not improved:
            break
    return w, float(best)


def rank_ids(block_ids, feats, w, question, qtype, tokens, run_text, promotion=True):
    """The original rank(): score, path boost on location questions, optional promotion. Returns (ordered ids, score vector)."""
    pred = feats @ np.asarray(w, dtype=np.float32)
    if is_loc(question, qtype):
        pred = pred + 1.0 * path_flags(block_ids, run_text)
    order = list(np.argsort(-pred))
    if promotion:
        costs = np.array([max(tokens.get(b, 1), 1) + 16 for b in block_ids], dtype=np.int32)
        order = promote(order, pred, costs)
    return [block_ids[i] for i in order], pred
