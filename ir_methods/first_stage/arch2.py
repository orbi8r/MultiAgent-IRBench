"""Interleaved compound first stage with no neural model and no fine tuned reranker.
Round 0 fuses the lexical and query side lists from arch1 pass 1 with RRF. Rounds 1 and 2 take the top 5 of the
current fused list as seeds and add: a feedback query (expansion terms of the seeds), a rare identifier query,
author and shift lists, a file graph, an outcome list (tool results that follow a seed in its conversation, up to 3
events) and an echo list (the first write of the text of a low novelty seed). Each round fuses everything so far.
Outputs a key free RRF list (arch2_rrf) and a small LambdaMART over the same lists (arch2_gbdt, 100 trees, 15
leaves), trained leave one run out on the other two runs' keys only.
    python arch2.py -> data/<run>/fam_arch2.json, arch2_time.json"""
import json, re, math, time
from collections import Counter
import numpy as np
from run_loader import RUNS
from contextual_pieces import toks
from bm25_tools import rank, rrf
from arch1 import Run, ident, fit, N, POOL

ROUNDS, SEEDS = 2, 5


def expand(r, q, seeds, sc):
    P, L = r.P, {}
    t = time.time()
    tf = Counter()
    for i in seeds:
        for w, c in Counter(r.tc[i]).items():
            tf[w] += c / len(r.tc[i]) * math.log(r.bc.n / r.bc.df[w])
    L["fb"] = rank(r.bc.scores(toks(q["question"]) + [w for w, _ in tf.most_common(20)]), N)
    idf = Counter()
    for i in seeds:
        for w in set(ident(P[i]["text"])):
            if r.br.df.get(w, 0) <= 30:
                idf[w] += 1
    L["idlink"] = rank(r.br.scores([w for w, _ in idf.most_common(15)]), N) if idf else []
    L["authshift"] = rrf(*[sorted(r.by_as[(P[i]["agent"], P[i]["shift"])], key=lambda j: -sc.get(j, 0)) for i in seeds[:3]])
    fe = []
    for i in seeds:
        for pth in r.F.files_of(P[i]["event"]):
            fe += r.path_events.get(pth, [])
    L["filegraph"] = sorted(dict.fromkeys(j for e in dict.fromkeys(fe) for j in r.by_event[e]), key=lambda j: -sc.get(j, 0))[:N]
    out = []
    for i in seeds:
        c, k = r.cpos[P[i]["event"]]
        for kk in range(k + 1, min(k + 4, len(r.conv[c]))):
            out += [j for j in r.by_event[r.conv[c][kk]] if P[j]["kind"] == "tool_result"]
    L["outcome"] = list(dict.fromkeys(out))
    fw = []
    for i in seeds:
        if r.nov[P[i]["id"]] < 0.5:
            w = re.findall(r"[a-z0-9]+", P[i]["text"].lower())
            fw += [r.first[h] for h in (hash(" ".join(w[k:k + 8])) for k in range(0, max(len(w) - 7, 0), 4)) if h in r.first]
    L["echo_origin"] = list(dict.fromkeys(fw))
    r.T["expand"] += time.time() - t
    return L


def main():
    t = time.time(); R = {n: Run(n) for n in RUNS}; tinit = time.time() - t
    rows = {}
    for n, r in R.items():
        for q in r.qs:
            L, sc = r.pass1(q)
            t = time.time(); cur = rrf(*L.values()); r.T["fuse"] += time.time() - t
            allL = dict(L)
            for rd in range(ROUNDS):
                E = expand(r, q, cur[:SEEDS], sc)
                allL.update({f"{k}{rd}": v for k, v in E.items()})
                t = time.time(); cur = rrf(*allL.values()); r.T["fuse"] += time.time() - t
            pool = list(dict.fromkeys(cur[:POOL] + [i for l in allL.values() for i in l[:100]]))
            rows[(n, q["qid"])] = (q, cur, pool, r.feats(q, pool, allL), r.labels(q, pool))
    out = {n: {"arch2_rrf": {}, "arch2_gbdt": {}} for n in RUNS}
    for test in RUNS:
        m = fit([(v[3], v[4]) for k, v in rows.items() if k[0] != test])
        for k, (q, cur, pool, X, y) in rows.items():
            if k[0] != test:
                continue
            r = R[test]; t = time.time(); s = m.predict(X); r.T["gbdt"] += time.time() - t
            out[test]["arch2_rrf"][k[1]] = [r.ids[i] for i in cur[:N]]
            out[test]["arch2_gbdt"][k[1]] = [r.ids[pool[i]] for i in np.argsort(-s)][:N]
    for n in RUNS:
        json.dump(out[n], open(f"data/{n}/fam_arch2.json", "w"))
    times = {n: {k: round(v, 2) for k, v in R[n].T.items()} for n in RUNS}
    times["init_all_runs"] = round(tinit, 1)
    json.dump(times, open("arch2_time.json", "w"), indent=1)
    print(json.dumps(times))


if __name__ == "__main__":
    main()
