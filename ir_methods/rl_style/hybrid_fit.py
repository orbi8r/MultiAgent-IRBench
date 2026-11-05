"""Refined hybrids: the original Model B machinery on OUR pool (fuse0 plus expansion), with the ftpub score from the pool step.
CPU only, reads $RL_STYLE_FEATS/pool_<run>.pkl (pool_features.py). Opens only dev answer keys. Everything fitted on the 3 dev runs (leave one run out for dev numbers, all 3 for the frozen test setting). Declared before any score:
  h1 pl_h1_dppool    the original 12 features refitted on the pool, the original coordinate ascent, n_iter 5, anchor init
  h2 pl_h2_dpft      h1 plus the ftpub z score as a 13th feature (per question z over the pool, clip [-3,3], init weight 0)
  h3 pl_h3_blend     z(h1 score) + w * z(ftpub); w from GRID by the dev BAR score of the training runs (no promotion rule)
  h4 pl_h4_seeds     our pipeline with ftpub order everywhere, except location questions: the original h1 order picks the provenance seeds
  each of h1 h2 h3 also as <tag>_lb: the same order passed through our list builder (earliest copy, location provenance list). The original early exit stop is dropped: the list format has no stop mechanism.
    python hybrid_fit.py OUT"""
import json, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

from rl_common import DEV, TEST, FEATS, load_key, write_sub, freeze, frozen, write_row, add_timing, limit, dev_table, fmt_dev
import score as S
from score import BUDGETS, read_order, shares
from file_provenance import Files
import build_list as FN
from hybrid_lib import her12, prep, fit_cd, rank_ids, is_loc, path_flags

OUT = Path(sys.argv[1])
N_ITER = 5
GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
INIT12 = [1.0] + [0.0] * 11
INIT13 = INIT12 + [0.0]
SEC = "S2 RL style refined hybrids (the original policies on our pool, dev only fits)"
TAGS = dict(h1="pl_h1_dppool", h2="pl_h2_dpft", h3="pl_h3_blend", h4="pl_h4_seeds")
VARIANTS = [("h1", False), ("h1", True), ("h2", False), ("h2", True), ("h3", False), ("h3", True), ("h4", True)]


def z(a):
    s = a.std()
    return np.zeros_like(a) if s < 1e-9 else np.clip((a - a.mean()) / s, -3, 3)


def tag_of(kind, lb):
    return TAGS[kind] + ("_lb" if lb and kind != "h4" else "")


ST = {}  # run -> dict(items, run, F, qs, fast12, fast13)


def load(r, with_key):
    items = pickle.load(open(FEATS / f"pool_{r}.pkl", "rb"))
    run = S.load_run(f"runs/{r}")
    blocks = [json.loads(l) for l in open(f"runs/{r}/blocks.jsonl", encoding="utf-8")]
    events = [json.loads(l) for l in open(f"runs/{r}/events.jsonl", encoding="utf-8")]
    F = Files(events, blocks)
    F.by_event = defaultdict(list)
    for p in blocks:
        F.by_event[p["event"]].append(p["id"])
    qs = {q["qid"]: q for q in (json.loads(l) for l in open(f"runs/{r}/questions.jsonl", encoding="utf-8"))}
    st = dict(items=items, run=run, F=F, qs=qs, fast12=[], fast13=[])
    for qid, it in items.items():
        it["x12"] = her12(it["block_ids"], it["retrieval_feats"], it["struct_feats"], run["tokens"])
        it["zft"] = z(it["ftpub"])
    if with_key:
        key = load_key(r)
        for qid, it in items.items():
            if qid not in key:
                continue
            d = dict(block_ids=it["block_ids"], question=it["question"], qtype=it["qtype"], tokens_map=run["tokens"],
                     credits=S.credits(key[qid], run), run_text=run["text"])
            for name, X in (("fast12", it["x12"]), ("fast13", np.hstack([it["x12"], it["zft"][:, None]]))):
                p = prep(d, X)
                if p:
                    p["zft"] = it["zft"]
                    st[name].append(p)
    ST[r] = st


def bar(it, order):
    ids = [it["block_ids"][i] for i in order]
    found = [[] for _ in it["atoms"]]
    for x, t in read_order(ids, it["tokens"], max(it["F"], BUDGETS[-1])):
        for f, a in zip(found, it["atoms"]):
            if a.get(x):
                f.append((t, a[x]))
    return shares(found, it["budgets_adj"])[0]


def blend_scores(x12, w1, wf, zft, has_path, loc):
    pred = x12 @ np.asarray(w1, dtype=np.float32)
    if loc:
        pred = pred + 1.0 * has_path
    return z(pred) + wf * zft


def pick_w(fast, w1):
    best, bw = -1.0, 0.0
    for wf in GRID:
        tot = 0.0
        for it in fast:
            s = blend_scores(it["feats"], w1, wf, it["zft"], it["has_path"], it["is_loc"])
            tot += bar(it, list(np.argsort(-s)))
        m = tot / len(fast)
        if m > best + 1e-6:  # ties keep the smaller weight
            best, bw = m, wf
    return bw, best


def fold_models(fold):
    """weights per kind for one fold, resumable per file"""
    tr = [r for r in DEV if r != fold]
    W = {}
    for kind, init, fastn in (("h1", INIT12, "fast12"), ("h2", INIT13, "fast13")):
        f = OUT / "weights" / f"{kind}_{fold}.json"
        if f.exists():
            W[kind] = json.load(open(f))
            continue
        t0 = time.time()
        fast = [x for r in tr for x in ST[r][fastn]]
        w, sc = fit_cd(fast, init, N_ITER)
        W[kind] = dict(fold=fold, train_runs=tr, init=init, n_iter=N_ITER, train_score=sc, weights=[float(v) for v in w])
        f.parent.mkdir(parents=True, exist_ok=True)
        json.dump(W[kind], open(f, "w"), indent=1)
        add_timing(OUT, TAGS[kind], fold, time.time() - t0)
        print(kind, fold, "train", round(sc, 4), round(time.time() - t0), "s", flush=True)
    f = OUT / "weights" / f"h3_{fold}.json"
    if f.exists():
        W["h3"] = json.load(open(f))
    else:
        fast = [x for r in tr for x in ST[r]["fast12"]]
        wf, sc = pick_w(fast, W["h1"]["weights"])
        W["h3"] = dict(fold=fold, train_runs=tr, grid=GRID, w=wf, train_score=sc, h1_weights=W["h1"]["weights"])
        json.dump(W["h3"], open(f, "w"), indent=1)
        print("h3", fold, "w", wf, "train", round(sc, 4), flush=True)
    return W


def make_subs(r, W):
    """-> {(kind, lb): sub} for run r under fitted settings W (from fold_models or the frozen files)"""
    st = ST[r]
    F, run = st["F"], st["run"]
    tok, text = run["tokens"], run["text"]
    out = {v: {} for v in VARIANTS}
    for qid, it in st["items"].items():
        q = st["qs"][qid]
        b, qt, question = it["block_ids"], it["qtype"], it["question"]
        ids1, _ = rank_ids(b, it["x12"], W["h1"]["weights"], question, qt, tok, text)
        x13 = np.hstack([it["x12"], it["zft"][:, None]])
        ids2, _ = rank_ids(b, x13, W["h2"]["weights"], question, qt, tok, text)
        loc = is_loc(question, qt)
        s3 = blend_scores(it["x12"], W["h1"]["weights"], W["h3"]["w"], it["zft"], path_flags(b, text) if loc else np.zeros(len(b), np.float32), loc)
        ids3 = [b[i] for i in np.argsort(-s3)]
        for kind, ids in (("h1", ids1), ("h2", ids2), ("h3", ids3)):
            out[(kind, False)][qid] = ids
            out[(kind, True)][qid] = FN.build(F, q, ids, [-k for k in range(len(ids))], text)
        # h4: ftpub everywhere, the original order only picks the location seeds
        ft = [b[i] for i in np.argsort(-it["ftpub"])]
        if qt == "location":
            seeds = FN.dedup(ids1, text)
            prov = FN.location_list(F, seeds)[:-len(seeds)] if seeds else []
            out[("h4", True)][qid] = list(dict.fromkeys(prov + FN.dedup(ft, text) + ft))[:400]
        else:
            out[("h4", True)][qid] = FN.build(F, q, ft, [-k for k in range(len(ft))], text)
    for v in out.values():
        for qid in limit(st["qs"].keys()):
            v.setdefault(qid, [])
    return out


for r in DEV:
    load(r, True)
folds = {f: fold_models(f) for f in DEV + ["all3"]}
for r in DEV:
    subs = make_subs(r, folds[r])
    for (kind, lb), sub in subs.items():
        write_sub(OUT, "dev", tag_of(kind, lb), r, sub)
# freeze every hybrid from the 3 dev runs before any test sub exists
for kind, lb in VARIANTS:
    t = tag_of(kind, lb)
    freeze(OUT, t, dict(declared=True, kind=kind, list_builder=lb, fit_runs=DEV, n_iter=N_ITER, grid=GRID, feature_z="per question z of ftpub over the pool, clip [-3,3]",
                        h1=folds["all3"]["h1"], h2=folds["all3"]["h2"], h3=folds["all3"]["h3"], builder=dict(M_CONTENT=FN.M_CONTENT, SCAN=FN.SCAN, EARLIEST=FN.EARLIEST),
                        pool="OURS 20_ftpub_pool ftpub_pool.json keys, ordered by the original fused score",
                        evidence="dev runs only; no setting chosen by a test number"))
for r in [t for t in TEST if t not in DEV]:
    load(r, False)
    fz = {v: frozen(OUT, tag_of(*v)) for v in VARIANTS}
    for (kind, lb), sub in make_subs(r, dict(h1=fz[("h1", False)]["h1"], h2=fz[("h2", False)]["h2"], h3=fz[("h3", False)]["h3"])).items():
        write_sub(OUT, "test", tag_of(kind, lb), r, sub)
    ST.pop(r)
META = {
    "h1": "the original 12 feature Model B refitted on our pool, coordinate ascent on the BAR metric",
    "h2": "the original Model B with the ftpub z score as a 13th feature",
    "h3": "fusion z(the original Model B score) + w z(ftpub), w chosen on dev",
    "h4": "our pipeline (ftpub order, list builder) with the original Model B choosing the location provenance seeds",
}
for kind, lb in VARIANTS:
    t = tag_of(kind, lb)
    res = dev_table(OUT, t, S)
    wtxt = ", ".join(f + "=" + str(folds[f]["h3"]["w"]) for f in folds)
    extra = " w per fold " + wtxt if kind == "h3" else ""
    write_row(OUT, t, section=SEC, method=META[kind] + (", through our list builder" if lb and kind != "h4" else ", raw order" if kind != "h4" else ""), doc_kind="LORO",
              frozen_setting=f"fit on 3 dev runs, n_iter {N_ITER}, anchor init" + (f", w grid {GRID}" if kind == "h3" else "") + ("" if kind != "h4" else ", seeds from h1 weights"),
              clean_reason="the original features unfitted design, weights fitted on 3 dev runs only, ftpub trained on public data only: clean",
              dev_note=fmt_dev(res) + " (leave one run out)" + extra,
              notes="declared before any score; the original early exit stop dropped (the list has no stop mechanism)")
print("hybrids done", flush=True)
