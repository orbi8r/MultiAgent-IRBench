"""Sectional rerank helpers for the full3 study, CPU only. A run is the list of candidates of each question with cached
reader scores plus the structure nodes of run_structure.py. Method C adds a node score to the block score, method D learns a
ranking from node aggregates. Answer keys are opened only for dev runs (labels and dev scoring), never for test runs."""
import json, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import study_helpers
from run_structure import load_run as load_struct, NODE_TYPES

KINDS = ["tool_call", "tool_result", "model", "prompt"]
QTYPES = ["finding", "location", "who", "history", "state", "failure_fix", "decision"]
SOFT = (0.0, 0.25, 0.5, 1.0)
HARD = (0.5, 1.0, 2.0)
CONFIGS = [("flat", "-", 0.0)] + [(t, m, v) for t in NODE_TYPES for m, g in (("soft", SOFT[1:]), ("hard", HARD)) for v in g]
LGB = [dict(n_estimators=100, num_leaves=7), dict(n_estimators=300, num_leaves=7), dict(n_estimators=150, num_leaves=15)]


class RunData:
    """cands {qid: [piece ids]}, sc {qid: np.array aligned}, qs, builder state, structure and for dev runs key and credits."""
    def __init__(self, name, cands, scores, dev):
        self.name = name
        qs, self.F, self.run, self.P = study_helpers.builder(name)
        self.S = load_struct(name)
        self.qs = [q for q in qs if q["qid"] in cands]
        self.cands = {q["qid"]: [x for x in cands[q["qid"]] if x in self.S.info] for q in self.qs}  # expansion lists can hold non pieces
        self.sc = {q["qid"]: np.array([scores[q["qid"]][x] for x in self.cands[q["qid"]]], dtype=float) for q in self.qs}
        self.dev, self.key, self.cred = dev, None, {}
        if dev:
            import score as SC
            self.SC = SC
            self.key = json.load(open(Path(os.environ["IRBENCH_RUNS"]) / name / "answer_key.json", encoding="utf-8"))
            self.cred = {q["qid"]: SC.credits(self.key[q["qid"]], self.run) for q in self.qs}
        self.feat = {q["qid"]: qfeat(self, q) for q in self.qs}

    def build(self, q, scores):
        return study_helpers.build(self.F, q, self.run, self.cands[q["qid"]], list(scores))

    def score(self, q, ids):
        qid = q["qid"]
        return self.SC.score_query(self.key[qid], ids, self.run, credit=self.cred[qid])["score"]


def qfeat(rd, q):
    """Per question: Z[type] z scored node score per candidate (0 without node), T3[type] in a top 3 node, X the matrix of D features."""
    qid = q["qid"]; c, s, n, S = rd.cands[qid], rd.sc[qid], len(rd.cands[qid]), rd.S
    order = np.argsort(-s, kind="stable"); rank = np.empty(n); rank[order] = np.arange(n)
    cols = [s, s - s.max(), (s - s.mean()) / (s.std() + 1e-9), rank, np.arange(n, dtype=float)]
    Z, T3 = {}, {}
    for t in NODE_TYPES:
        grp = defaultdict(list)
        for i, pid in enumerate(c):
            for nid in S.nodes_of(pid)[t]:
                grp[nid].append(i)
        agg = {}
        for nid, idx in grp.items():
            v = np.sort(s[idx])[::-1]
            agg[nid] = (float(v[0]), float(v[:2].mean()), len(idx))
        nrank = {nid: r for r, nid in enumerate(sorted(agg, key=lambda k: (-agg[k][1], k)))}
        top3 = set(sorted(agg, key=lambda k: (-agg[k][1], k))[:3])
        nmax, nm2, cnt, rin, nrk = (np.full(n, np.nan) for _ in range(5))
        t3 = np.zeros(n)
        for i, pid in enumerate(c):
            ids = S.nodes_of(pid)[t]
            if not ids:
                continue
            b = max(ids, key=lambda k: (agg[k][1], k))  # a piece with several files takes its best node
            nmax[i], nm2[i], cnt[i], nrk[i] = agg[b][0], agg[b][1], agg[b][2], nrank[b]
            rin[i] = float((s[grp[b]] > s[i]).sum())
            t3[i] = float(any(k in top3 for k in ids))
        ok = ~np.isnan(nm2)
        z = np.zeros(n)
        if ok.sum() > 1:
            z[ok] = (nm2[ok] - nm2[ok].mean()) / (nm2[ok].std() + 1e-9)
        Z[t], T3[t] = z, t3
        cols += [nmax, nm2, cnt, rin, nrk, s - nmax]
    info = [S.info[p] for p in c]
    kd = {p["id"]: p["kind"] for p in rd.P}
    cols += [np.array([KINDS.index(kd[p]) if kd[p] in KINDS else len(KINDS) for p in c], dtype=float),
             np.log1p([rd.run["tokens"][p] for p in c]), np.array([i["j"] / i["nsib"] for i in info]),
             np.array([i["nsib"] for i in info], dtype=float), np.array([i["pos"] for i in info]),
             np.full(n, float(QTYPES.index(q["qtype"]) if q["qtype"] in QTYPES else len(QTYPES)))]
    return dict(Z=Z, T3=T3, X=np.column_stack(cols))


def c_scores(rd, qid, cfg):
    t, mode, v = cfg
    s = rd.sc[qid]
    if t == "flat":
        return s
    f = rd.feat[qid]
    return s + v * (f["Z"][t] if mode == "soft" else f["T3"][t])


def labels(rd, qid):
    """0 to 3 from the share of the answer atoms a candidate states (dev keys)."""
    cr = rd.cred[qid]
    return np.array([min(3, math.ceil(3 * sum(1 for a in cr if p in a) / max(len(cr), 1))) for p in rd.cands[qid]])


def eval_c(rd, cfg):
    """{qid: score} of method C with config cfg on a dev run."""
    return {q["qid"]: rd.score(q, rd.build(q, c_scores(rd, q["qid"], cfg))) for q in rd.qs}


def pooled(res, runs):
    v = [x for r in runs for x in res[r].values()]
    return sum(v) / len(v)


def pick_c(table, train, types=None):
    """Config with the best pooled score over the train runs; ties go to the earlier (simpler) config."""
    cs = [c for c in CONFIGS if types is None or c[0] in types or c[0] == "flat"]
    return max(cs, key=lambda c: (round(pooled(table[c], train), 12), -cs.index(c)))


def fit_d(rds, hp):
    import lightgbm as lgb
    X = np.vstack([r.feat[q["qid"]]["X"] for r in rds for q in r.qs])
    y = np.concatenate([labels(r, q["qid"]) for r in rds for q in r.qs])
    g = [len(r.cands[q["qid"]]) for r in rds for q in r.qs]
    m = lgb.LGBMRanker(objective="lambdarank", learning_rate=0.05, min_child_samples=20, colsample_bytree=0.8, reg_lambda=1.0,
                       random_state=0, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1, **hp)
    return m.fit(X, y, group=g)


def eval_d(model, rd):
    return {q["qid"]: rd.score(q, rd.build(q, model.predict(rd.feat[q["qid"]]["X"]))) for q in rd.qs}


def pick_d(train):
    """Hyperparameters by leave one run out inside the train runs (one run only: train and eval on it, smoke)."""
    best, bs = 0, -1
    for k, hp in enumerate(LGB):
        res = {}
        for r in train:
            rest = [x for x in train if x is not r] or [r]
            res[r.name] = eval_d(fit_d(rest, hp), r)
        s = pooled(res, [r.name for r in train])
        if s > bs + 1e-12:
            best, bs = k, s
    return best
