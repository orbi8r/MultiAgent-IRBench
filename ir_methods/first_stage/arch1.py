"""Algorithmic first stage with two small boosted tree selectors and no neural model. Each stage is timed.
Pass 1 builds lexical and query side lists (BM25 over context, plain text and raw identifiers, novelty pruned text,
clause split, question type expansion, feedback expansion) into a pool, and GBDT1 ranks the pool. Pass 2 takes the
top 5 seeds from GBDT1 and builds structure lists (conversation neighbours, rare identifier links, file graph, author
and shift, first write of the seed text), and GBDT2 ranks the final candidates over pass 1 and pass 2 features.
Both GBDTs (lightgbm LambdaMART, 100 trees, 15 leaves) are trained leave one run out on the other two runs' answer
keys only, so nothing is fit on the held out run.
    python arch1.py -> data/<run>/fam_arch.json {arch1, arch1_pass1}, arch1_time.json"""
import json, re, math, time
from collections import Counter, defaultdict
import numpy as np, lightgbm as lgb
from run_loader import load, S, RUNS, words
from contextual_pieces import toks
from bm25_tools import BM, rank, rrf
from novelty import novelty
from file_provenance import Files

N, POOL = 1000, 300
QX = {"who": "git commit author committed shift", "location": "mv moved file path saved folder", "history": "first earlier originally"}
ID = re.compile(r"#?\w*\d\w*|\w+[_./-]\w[\w./-]*")
KINDS = ["tool_call", "tool_result", "model", "prompt", "user"]
QT = ("who", "location", "finding", "history")


def ident(text):
    out = []
    for m in ID.findall(text):
        m = m.lower().lstrip("#")
        if len(m) >= 3:
            out.append(m)
            parts = re.split(r"([-_./])", m)
            out += ["".join(parts[:k]) for k in range(3, len(parts) - 1, 2)]
    return out


def clauses(q):
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+(?=in which|what|who|where|when|how|which)|\bso that\b|\bafter\b|\bbefore\b", q)
    return [p for p in parts if len(p.split()) >= 3]


class Run:
    def __init__(self, name):
        T = self.T = defaultdict(float)
        d, run, key, qs, blocks, events = load(name)
        self.name, self.run, self.key, self.qs = name, run, key, qs
        P = self.P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
        self.ids = [p["id"] for p in P]
        t = time.time(); tc, tp = [toks(p["ctx"]) for p in P], [toks(p["text"]) for p in P]
        traw = [words(p["text"]) + ident(p["text"]) for p in P]; T["tokenize"] += time.time() - t
        t = time.time(); self.nov = novelty(P); keep = self.keep = [i for i, p in enumerate(P) if self.nov[p["id"]] > 0.2]; T["novelty"] += time.time() - t
        t = time.time()
        self.bc, self.bp, self.br = BM(tc), BM(tp), BM(traw)
        self.nc, self.np_ = BM([tc[i] for i in keep]), BM([tp[i] for i in keep]); T["index"] += time.time() - t
        self.tc = tc
        t = time.time()
        conv = defaultdict(list)
        for e in events:
            conv[e["conv"]].append(e["id"])
        self.conv, self.cpos = conv, {eid: (c, k) for c, l in conv.items() for k, eid in enumerate(l)}
        self.by_event = defaultdict(list)
        for i, p in enumerate(P):
            self.by_event[p["event"]].append(i)
        self.F = Files(events, P); self.path_events = defaultdict(list)
        for eid, ps in self.F.paths.items():
            for pth in ps:
                self.path_events[pth].append(eid)
        self.by_as = defaultdict(list)
        for i, p in enumerate(P):
            self.by_as[(p["agent"], p["shift"])].append(i)
        self.first = {}
        for i, p in enumerate(P):
            w = re.findall(r"[a-z0-9]+", p["text"].lower())
            for k in range(0, max(len(w) - 7, 0), 4):
                self.first.setdefault(hash(" ".join(w[k:k + 8])), i)
        T["structure_index"] += time.time() - t

    def pass1(self, q):
        t = time.time(); qt = toks(q["question"])
        sc = self.bc.scores(qt); lc, lp = rank(sc, N), rank(self.bp.scores(qt), N); base = rrf(lc, lp)
        L = {"bm_ctx": lc, "bm_plain": lp, "bm_raw": rank(self.br.scores(words(q["question"]) + ident(q["question"])), N),
             "bm_nov": [self.keep[i] for i in rrf(rank(self.nc.scores(qt), N), rank(self.np_.scores(qt), N))]}
        tf = Counter()
        for i in base[:10]:
            for w, c in Counter(self.tc[i]).items():
                tf[w] += c / len(self.tc[i]) * math.log(self.bc.n / self.bc.df[w])
        L["rm3"] = rank(self.bc.scores(qt + [w for w, _ in tf.most_common(25)]), N)
        cl = clauses(q["question"])
        L["qdecomp"] = rrf(*[rank(self.bc.scores(toks(c)), N) for c in cl]) if len(cl) > 1 else base
        L["qtype"] = rank(self.bc.scores(qt + toks(QX.get(q["qtype"], ""))), N)
        self.T["pass1_lists"] += time.time() - t
        return L, sc

    def pass2(self, seeds, sc):
        t = time.time(); P = self.P; L = {}
        nb = []
        for dist in range(4):
            for i in seeds:
                c, k = self.cpos[P[i]["event"]]
                for kk in {k - dist, k + dist}:
                    if 0 <= kk < len(self.conv[c]):
                        nb += self.by_event[self.conv[c][kk]]
        L["nbr"] = list(dict.fromkeys(nb))[:N]
        idf = Counter()
        for i in seeds:
            for w in set(ident(P[i]["text"])):
                if self.br.df.get(w, 0) <= 30:
                    idf[w] += 1
        L["idlink"] = rank(self.br.scores([w for w, _ in idf.most_common(15)]), N) if idf else []
        fe = []
        for i in seeds:
            for pth in self.F.files_of(P[i]["event"]):
                fe += self.path_events.get(pth, [])
        L["filegraph"] = list(dict.fromkeys(j for e in dict.fromkeys(fe) for j in self.by_event[e]))[:N]
        L["authshift"] = rrf(*[sorted(self.by_as[(P[i]["agent"], P[i]["shift"])], key=lambda j: -sc.get(j, 0)) for i in seeds[:3]])
        fw = []
        for i in seeds:
            w = re.findall(r"[a-z0-9]+", P[i]["text"].lower())
            fw += [self.first[h] for h in (hash(" ".join(w[k:k + 8])) for k in range(0, max(len(w) - 7, 0), 4)) if h in self.first]
        L["firstwrite"] = list(dict.fromkeys(fw))[:N]
        self.T["pass2_lists"] += time.time() - t
        return L

    def feats(self, q, pool, lists, extra=None):
        R = {f: {x: r for r, x in enumerate(l)} for f, l in lists.items()}
        X = []
        for i in pool:
            p = self.P[i]
            X.append([1 / (60 + R[f][i]) if i in R[f] else 0 for f in sorted(lists)] + [math.log(1 + p["tokens"]), self.nov[p["id"]],
                     p["agent"] == q["asker"]] + [p["kind"] == k for k in KINDS] + [q["qtype"] == t for t in QT] + (extra or []))
        return np.array(X, dtype=np.float32)

    def labels(self, q, pool):
        tok = self.run["tokens"]; lab = defaultdict(int)
        for a in S.credits(self.key[q["qid"]], self.run):
            ch = min(tok[x] for x in a)
            for x in a:
                lab[x] = max(lab[x], 2 if tok[x] <= 1.5 * ch + 50 else 1)
        return np.array([lab[self.ids[i]] for i in pool])


def fit(rows):
    m = lgb.LGBMRanker(n_estimators=100, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1)
    m.fit(np.vstack([r[0] for r in rows]), np.concatenate([r[1] for r in rows]), group=[len(r[0]) for r in rows])
    return m



def main():
    R = {n: Run(n) for n in RUNS}
    P1 = {}
    for n, r in R.items():
        for q in r.qs:
            L, sc = r.pass1(q)
            pool = list(dict.fromkeys(i for l in L.values() for i in l[:POOL]))
            P1[(n, q["qid"])] = (q, L, sc, pool, r.feats(q, pool, L), r.labels(q, pool))
    out = {n: {"arch1_pass1": {}, "arch1": {}} for n in RUNS}
    for test in RUNS:
        tr = [n for n in RUNS if n != test]
        m1 = fit([(v[4], v[5]) for k, v in P1.items() if k[0] in tr])
        P2 = {}
        for k, (q, L, sc, pool, X, y) in P1.items():
            r = R[k[0]]
            t = time.time(); s1 = m1.predict(X); order = [pool[i] for i in np.argsort(-s1)]
            if k[0] == test:
                r.T["gbdt1"] += time.time() - t
            L2 = r.pass2(order[:5], sc) if k[0] == test or True else None
            pool2 = list(dict.fromkeys(order[:POOL] + [i for l in L2.values() for i in l[:100]]))
            s1d = dict(zip(pool, s1))
            X2 = r.feats(q, pool2, {**L, **L2}, None)
            X2 = np.hstack([X2, np.array([[s1d.get(i, -10)] for i in pool2], dtype=np.float32)])
            P2[k] = (pool2, X2, r.labels(q, pool2), order)
        m2 = fit([(v[1], v[2]) for k, v in P2.items() if k[0] in tr])
        for k, (pool2, X2, y2, order) in P2.items():
            if k[0] != test:
                continue
            r = R[test]; t = time.time(); s2 = m2.predict(X2); r.T["gbdt2"] += time.time() - t
            out[test]["arch1_pass1"][k[1]] = [r.ids[i] for i in order[:N]]
            out[test]["arch1"][k[1]] = [r.ids[i] for i in np.argsort(-s2).tolist() for i in [pool2[i]]][:N]
    for n in RUNS:
        json.dump(out[n], open(f"data/{n}/fam_arch.json", "w"))
    # pass 2 lists are rebuilt in each of the 3 folds, so divide their time by 3
    times = {n: {k: round(v / (3 if k == "pass2_lists" else 1), 2) for k, v in R[n].T.items()} for n in RUNS}
    json.dump(times, open("arch1_time.json", "w"), indent=1)
    print(json.dumps(times))


if __name__ == "__main__":
    main()
