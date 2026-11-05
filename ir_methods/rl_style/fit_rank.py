"""P0 to P3: first stage lists, rules only, Model B fitted on dev (LORO, all3 for test), shipped checkpoint probe.
Opens only dev answer keys (asserted in rl_common). Writes $IRBENCH_OUT/{subs,frozen,rows,devscore,weights,timing}.
    python fit_rank.py OUT"""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np

from rl_common import (DEV, TEST, DEDUPS, FEATS, load_key, write_sub, freeze, frozen, write_row, add_timing, dev_table, fmt_dev)
import score as S
from method.reranker import SubmodularDirectPolicy

OUT = Path(sys.argv[1])
N_ITER = 5  # the original training script's default, fixed, never tuned
INIT = [1.0] + [0.0] * 11  # first stage order only, carries no dev information
SHIPPED = Path("method/checkpoints/direct_policy_master.json")
SEC = "S1 RL style reruns (one metric, dev only fits)"
_runs = {}


def side(r, dd):
    if (r, dd) not in _runs:
        _runs[(r, dd)] = (pickle.load(open(FEATS / f"items_{r}_{dd}.pkl", "rb")), S.load_run(f"runs/{r}"))
    return _runs[(r, dd)]


def dev_items(r, dd):
    items, run = side(r, dd)
    key = load_key(r)  # asserts r is a dev run
    return [dict(it, tokens_map=run["tokens"], answer_entry=key[q], credits=S.credits(key[q], run), run_text=run["text"])
            for q, it in items.items() if q in key]


def rank(policy, r, dd):
    items, run = side(r, dd)
    sub = {q: policy.rank(it["block_ids"], it["retrieval_feats"], it["struct_feats"], run["tokens"], question=it["question"],
                          qtype=it["qtype"], run_text=run["text"]) for q, it in items.items()}
    for q in json.load(open(FEATS / f"fs_{r}_{dd}.json")):
        sub.setdefault(q, [])
    return sub


def pol(w):
    return SubmodularDirectPolicy(weights=np.array(w, dtype=np.float32))  # explicit weights: never loads a checkpoint


for dd in DEDUPS:
    tags = {k: f"pl_{k}_{dd}" for k in ("fs", "rules", "dp", "shipped")}
    fsd = lambda r: json.load(open(FEATS / f"fs_{r}_{dd}.json"))
    # declared, no choice: first stage, rules, shipped
    freeze(OUT, tags["fs"], dict(declared=True, choice="none", dedup=dd, stage="the original first stage lists, no learning"))
    freeze(OUT, tags["rules"], dict(declared=True, choice="none", dedup=dd, init=INIT, note="anchor only weights plus the original hard coded location path boost and 125 token promotion"))
    shipped_w = json.load(open(SHIPPED))["weights"]
    freeze(OUT, tags["shipped"], dict(declared=True, choice="none", dedup=dd, weights=shipped_w, sha256=hashlib.sha256(open(SHIPPED, "rb").read()).hexdigest(),
                                      note="the original committed direct_policy_master.json, training data not recoverable from the commit"))
    for r in DEV:
        write_sub(OUT, "dev", tags["fs"], r, fsd(r))
        write_sub(OUT, "dev", tags["rules"], r, rank(pol(INIT), r, dd))
        write_sub(OUT, "dev", tags["shipped"], r, rank(pol(shipped_w), r, dd))
    # fits: one per left out dev run, then all dev
    cache = {r: dev_items(r, dd) for r in DEV}
    final = {}
    for fold in DEV + ["all3"]:
        wf = OUT / "weights" / f"dp_{dd}_{fold}.json"
        tr = [r for r in DEV if r != fold]
        if wf.exists():  # resume marker
            final[fold] = pol(json.load(open(wf))["weights"])
            continue
        t0 = time.time()
        p = pol(INIT)
        train_score = p.fit([it for r in tr for it in cache[r]], n_iter=N_ITER, verbose=False)
        final[fold] = p
        wf.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(fold=fold, train_runs=tr, dedup=dd, init=INIT, n_iter=N_ITER, train_score=float(train_score),
                       weights=[float(x) for x in p.weights], names=list(p.FEATURE_NAMES)), open(wf, "w"), indent=1)
        add_timing(OUT, tags["dp"], fold, time.time() - t0)
        print(dd, fold, "train", round(float(train_score), 4), round(time.time() - t0), "s", flush=True)
    for r in DEV:
        write_sub(OUT, "dev", tags["dp"], r, rank(final[r], r, dd))
    # freeze the dp setting (weights fitted on the 3 dev runs only) before any test sub of any tag
    all3 = json.load(open(OUT / "weights" / f"dp_{dd}_all3.json"))
    freeze(OUT, tags["dp"], dict(declared=True, dedup=dd, init=INIT, n_iter=N_ITER, fit_runs=DEV, weights=all3["weights"],
                                 evidence="coordinate ascent on the BAR metric of the 3 dev runs, no selection among settings; dev numbers are leave one run out folds"))
    fz = {k: frozen(OUT, t) for k, t in tags.items()}
    for r in TEST:
        write_sub(OUT, "test", tags["fs"], r, fsd(r))
        write_sub(OUT, "test", tags["rules"], r, rank(pol(fz["rules"]["init"]), r, dd))
        write_sub(OUT, "test", tags["shipped"], r, rank(pol(fz["shipped"]["weights"]), r, dd))
        write_sub(OUT, "test", tags["dp"], r, rank(pol(fz["dp"]["weights"]), r, dd))
    meta = {
        "fs": ("P0 RL style first stage (BM25, field BM25, MiniLM dense, RRF, kind multipliers, moves, neighbours)", "FIXED",
               "no learning, the original A1 constants", "A1 constants were fitted in the old era on an earlier key set that included the test runs: ambiguous"),
        "rules": ("P1 RL style first stage plus hard rules (location path boost +1.0, 125 token promotion), anchor weights only", "FIXED",
                  "weights [1,0,...,0]", "same old era constants as P0 plus hand rules: ambiguous"),
        "dp": ("P2 RL style Model B, 12 feature linear policy, coordinate ascent on the BAR metric", "LORO",
               f"anchor init, n_iter {N_ITER}, fit on 3 dev runs", "weights fitted on dev only; first stage design and A1 constants came from the old era: design exposed"),
        "shipped": ("P3 RL style committed direct_policy_master.json as a probe", "PROBE",
                    "the original committed weights, no refit", "checkpoint training data not recoverable from the commit, dev score is in sample: flag"),
    }
    for k, t in tags.items():
        res = dev_table(OUT, t, S)
        m = meta[k]
        write_row(OUT, t, section=SEC, method=m[0] + f", dedup {dd}", doc_kind=m[1], frozen_setting=m[2] + f", dedup {dd}", clean_reason=m[3],
                  dev_note=fmt_dev(res) + (" (in sample)" if k == "shipped" else " (leave one run out)" if k == "dp" else ""),
                  notes="the original claim 0.4485 test main for exact dp; this row is the same method under our protocol (keys of 3 dev runs only, v5_abs scorer)")
