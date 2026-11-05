"""python sectional_rerank.py 06|4b: sectional rerank (method C node score added to the block score, method D LightGBM lambdarank on node
aggregates) over cached reader scores. Dev numbers are leave one run out over the dev runs; the one config per method frozen
from all dev runs scores the test runs once. 06 = z2 K200 with ftpub 0.6B (gpu_z2s.json), 4b = the 4B cache (rc/ftpub4b2x).
Env: IRBENCH_WORK IRBENCH_OUT DEVR TESTR NQ RC, as set by the shell library."""
import json, os, sys, time
import study_helpers, control_tags
import sectional_lib as L

mode = sys.argv[1]
DEV, TEST = study_helpers.DEVR, study_helpers.TESTR
SEC = {"06": "10.1 full3 sectional on cached ftpub 0.6B scores (z2 K200)", "4b": "10.2 full3 sectional on cached 4B scores (ftpub4b2x)"}[mode]


def inputs06(run):
    G = json.load(open(f"data/{run}/gpu_z2s.json"))
    m = [k for k in G["scores"] if k.endswith("+lora=models/ftpub")]
    assert len(m) == 1, list(G["scores"])
    return G["cands"], {q: dict(zip(G["cands"][q], v)) for q, v in G["scores"][m[0]].items()}


def inputs4b(run, exp):
    import yes_no_scorer
    p = next(x for x in (f"{study_helpers.BASE}/rc/ftpub4b2x/{run}.json", f"{study_helpers.BASE}/rc3/ftpub4b2x/{run}.json", f"{os.environ.get('RC', '')}/ftpub4b2x/{run}.json") if os.path.exists(x))
    d = yes_no_scorer.Cache(p).d
    f1 = json.load(open(f"data/{run}/f1.json"))["fuse0"]
    first = {q: [x for x in f1[q] if x in d[q]] for q in d}
    cands = {q: first[q] + ([x for x in d[q] if x not in set(first[q])] if exp else []) for q in d}
    return cands, d


def bases():
    if mode == "06":
        return [("f3_sect06", "", inputs06, True)]
    return [("f3_sect4b", "", lambda r: inputs4b(r, False), False), ("f3_sect4b", "_exp", lambda r: inputs4b(r, True), False)]


def sub_of(rd, fn):
    return {q["qid"]: rd.build(q, fn(q["qid"])) for q in rd.qs}


GATES = {}
NOTE = "cached reader scores, CPU only, zero GPU"


def mean(v):
    v = list(v)
    return sum(v) / len(v)


def do_base(pfx, suf, loader, per_type):
    t0 = time.time()
    dev = [L.RunData(r, *loader(r), True) for r in DEV]
    dn = [r.name for r in dev]
    table = {c: {r.name: L.eval_c(r, c) for r in dev} for c in L.CONFIGS}
    study_helpers.log(pfx + suf, "table done", round(time.time() - t0), "s", len(L.CONFIGS), "configs")
    folds = {r.name: ([x for x in dn if x != r.name] or [r.name]) for r in dev}
    res = {}  # tag -> {run: {qid: score}} leave one run out dev scores
    tags = {}  # tag -> node types the leave one run out choice may use (None = all)
    if per_type:
        for t in L.NODE_TYPES:
            tags[f"{pfx}_c_{t}{suf}"] = [t]
    tags[f"{pfx}_c_best{suf}"] = None
    for tag, types in tags.items():
        res[tag] = {}
        for r in dev:
            cfg = L.pick_c(table, folds[r.name], types)
            res[tag][r.name] = table[cfg][r.name]
            study_helpers.put(tag, "dev", r.name, sub_of(r, lambda q: L.c_scores(r, q, cfg)), r.qs)
            study_helpers.log(tag, r.name, "held out, config", cfg, round(mean(res[tag][r.name].values()), 4))
    dtag = f"{pfx}_d{suf}"
    res[dtag] = {}
    for r in dev:
        tr = [x for x in dev if x.name != r.name] or [r]
        hp = L.LGB[L.pick_d(tr)]
        m = L.fit_d(tr, hp)
        res[dtag][r.name] = L.eval_d(m, r)
        study_helpers.put(dtag, "dev", r.name, sub_of(r, lambda q: m.predict(r.feat[q]["X"])), r.qs)
        study_helpers.log(dtag, r.name, "held out, hp", hp, round(mean(res[dtag][r.name].values()), 4))
    # gate: dev pooled over questions against the flat control, and not worse on 2 of 3 runs
    flat = table[L.CONFIGS[0]]
    ctrl = control_tags.control(f"{pfx}_c_best{suf}")[0]
    cf = f"{study_helpers.BASE}/ctrl_scores/dev/{ctrl}.json"
    gate = dict(control=ctrl, flat_pooled=L.pooled(flat, dn), flat_runs={n: mean(flat[n].values()) for n in dn})
    if os.path.exists(cf):
        pq = {r: v["per_question"] for r, v in json.load(open(cf))["runs"].items() if v}
        gate["flat_vs_control_maxdiff"] = max(abs(flat[n][q] - pq[n][q][1]) for n in dn if n in pq for q in flat[n] if q in pq[n])
    need = min(2, len(dn))
    for name, tag in (("c", f"{pfx}_c_best{suf}"), ("d", dtag)):
        runs = {n: mean(res[tag][n].values()) - mean(flat[n].values()) for n in dn}
        gain = L.pooled(res[tag], dn) - gate["flat_pooled"]
        gate[name] = dict(tag=tag, dev_pooled=L.pooled(res[tag], dn), gain=gain, run_gains=runs,
                          go=bool(gain >= 0.005 and sum(1 for v in runs.values() if v >= 0) >= need))
    gate["go"] = gate["c"]["go"] or gate["d"]["go"]
    # freeze from all dev runs, then the test runs once
    cfg = L.pick_c(table, dn)
    frz = study_helpers.freeze(f"{pfx}_c_best{suf}", dict(method="C", config=list(cfg), grid="flat, then each node type (event file as sec) soft 0.25 0.5 1 and hard 0.5 1 2", builder="build_list.build SCAN 30 EARLIEST"),
                    f"pooled dev over {dn}: best config {cfg} {L.pooled(table[cfg], dn):.4f}, flat {gate['flat_pooled']:.4f}; leave one run out {gate['c']['dev_pooled']:.4f}")
    cfg = tuple(frz["config"])
    k = L.pick_d(dev)
    frd = study_helpers.freeze(dtag, dict(method="D", lightgbm=L.LGB[k], lambdarank_labels="ceil(3 * share of atoms stated), dev keys", builder="build_list.build SCAN 30 EARLIEST"),
                    f"inner leave one run out over {dn}: hp {L.LGB[k]}; leave one run out dev pooled {gate['d']['dev_pooled']:.4f}")
    model = L.fit_d(dev, frd["lightgbm"])
    for tag, kind, what, fz in ((f"{pfx}_c_best{suf}", "FIXED", f"method C, config {cfg}", frz), (dtag, "LORO", f"method D, lightgbm {frd['lightgbm']}", frd)):
        study_helpers.row(tag, SEC, f"sectional rerank {what}, frozen from the dev runs", kind, json.dumps(fz), "dev subs are leave one run out over the dev runs", NOTE)
    for t, types in tags.items():
        if types:
            study_helpers.row(t, SEC, f"sectional rerank method C, node type {types[0]} only, config chosen leave one run out", "FIXED", "dev only, no test sub", "dev subs are leave one run out", NOTE)
    del dev
    for r in TEST:
        todo = [(f"{pfx}_c_best{suf}", "c"), (dtag, "d")]
        todo = [x for x in todo if not study_helpers.have(x[0], "test", r)]
        if not todo:
            continue
        ts = time.time()
        rd = L.RunData(r, *loader(r), False)
        for tag, kind in todo:
            sub = sub_of(rd, (lambda q: L.c_scores(rd, q, cfg)) if kind == "c" else (lambda q: model.predict(rd.feat[q]["X"])))
            study_helpers.put(tag, "test", r, sub, rd.qs)
            study_helpers.timing(tag, r, time.time() - ts, 0.0, cpu=True)  # features plus list building
        study_helpers.log("test", r, [t for t, _ in todo], round(time.time() - ts, 1), "s")
    GATES[suf.strip("_") or mode] = gate
    study_helpers.atomic(study_helpers.OUT / "gate.json", dict(GATES, go=any(v["go"] for v in GATES.values())), indent=1)
    return gate


for b in bases():
    do_base(*b)
g = json.load(open(study_helpers.OUT / "gate.json"))
g["why"] = "go when a method beats its flat control by 0.005 pooled over dev questions and is not worse on 2 of 3 dev runs: " + "; ".join(
    f"{k} c {v['c']['gain']:+.4f} d {v['d']['gain']:+.4f}" for k, v in g.items() if isinstance(v, dict))
study_helpers.atomic(study_helpers.OUT / "gate.json", g, indent=1)
print(json.dumps(g)[:1500])
