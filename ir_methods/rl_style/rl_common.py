"""Shared helpers for the rl_style scripts. cwd of every script is the repo root (it holds runs/ and the method/ package).
Test answer keys are never opened here: only dev runs are allowed to load a key (asserted)."""
import json, os, sys
from pathlib import Path

sys.path.insert(0, ".")
from run_loader import PUB

REAL_DEV = ["online_shop", "library_maintainers", "literature_review"]
REAL_TEST = ["on_call", "due_diligence", "executive_assistants", "help_desk", "textbook_team"]
DEV = os.environ.get("RL_STYLE_DEV", ",".join(REAL_DEV)).split(",")      # smoke runs shrink these
TEST = os.environ.get("RL_STYLE_TEST", ",".join(REAL_TEST)).split(",")
assert set(DEV) <= set(REAL_DEV) and set(TEST) <= set(REAL_TEST), "dev and test lists must stay inside the public split"
SMOKE_Q = int(os.environ.get("SMOKE_Q", "0"))                          # 0 = all questions
DEDUPS = ["exact", "mh85"]
FEATS = Path(os.environ.get("RL_STYLE_FEATS", "feats"))                  # key free feature pickles


def load_key(run):
    assert run in REAL_DEV, f"refusing to read an answer key for {run}"
    return json.load(open(f"{PUB}/{run}/answer_key.json", encoding="utf-8"))


def limit(qids):
    qids = list(qids)
    return qids[:SMOKE_Q] if SMOKE_Q else qids


def write_sub(out, split, tag, run, sub):
    p = Path(out) / "subs" / split / tag
    p.mkdir(parents=True, exist_ok=True)
    json.dump(sub, open(p / f"{run}.json", "w"))


def freeze(out, tag, settings):
    """Frozen setting file, written BEFORE any test sub of this tag. Test code reads its settings back from it."""
    p = Path(out) / "frozen" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(settings, open(p, "w"), indent=1)
    return p


def frozen(out, tag):
    return json.load(open(Path(out) / "frozen" / f"{tag}.json"))


def write_row(out, tag, **kw):
    p = Path(out) / "rows" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tag=tag, **kw), open(p, "w"), indent=1)


def add_timing(out, tag, run, secs, peak_gb=0.0):
    p = Path(out) / "timing" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = json.load(open(p)) if p.exists() else {}
    d[run] = dict(secs=round(secs, 1), peak_gb=round(peak_gb, 3))
    json.dump(d, open(p, "w"), indent=1)


def peak_gb():
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def score_dev(run, sub, S, cache={}):
    """repo score.py v5_abs on a dev run (dev keys only). Returns (main score, equal credit score, by_qtype)."""
    assert run in REAL_DEV
    if run not in cache:
        cache[run] = (load_key(run), S.load_run(f"{PUB}/{run}"))
    key, rn = cache[run]
    key = {q: e for q, e in key.items() if (not SMOKE_Q) or q in sub}
    main = S.report_for(key, {q: S.score_query(e, sub.get(q, []), rn) for q, e in key.items()})
    eq = S.report_for(key, {q: S.score_query_equal(e, sub.get(q, []), rn) for q, e in key.items()}, main=False)
    return main["overall"]["score"], eq["overall"]["score"], len(key)


def dev_table(out, tag, S, subs_dir=None):
    """Score all dev subs of a tag and write out/devscore/<tag>.json (per run main and equal credit, run mean)."""
    d = Path(subs_dir or Path(out) / "subs" / "dev") / tag
    res = {}
    for r in DEV:
        f = d / f"{r}.json"
        if f.exists():
            m, e, n = score_dev(r, json.load(open(f)), S)
            res[r] = dict(score=m, equal=e, n=n)
    if res:
        res["mean"] = sum(v["score"] for v in res.values()) / len(res)
    p = Path(out) / "devscore" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p, "w"), indent=1)
    return res


def fmt_dev(res):
    return "dev " + " ".join(f"{r}={v['score']:.4f}" for r, v in res.items() if r != "mean") + f" mean={res.get('mean', 0):.4f}"
