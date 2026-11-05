"""BM25 index, rank and RRF helpers for the first stage scripts and a comparison of plain text against contextualised
pieces (with split identifiers) on lexical score and recall. Answer keys are used for scoring only. Prints one line per
run and variant with the mean score and the mean recall at 10, 200 and 1000.
    python bm25_tools.py"""
import math, os, statistics as st
from collections import Counter, defaultdict
from run_loader import load, RUNS, S
from contextual_pieces import build, toks


class BM:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.len = [sum(d.values()) or 1 for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        self.df = Counter(w for d in self.docs for w in d)
        self.post = defaultdict(list)
        for i, d in enumerate(self.docs):
            for w in d:
                self.post[w].append(i)
        self.n, self.k1, self.b = len(docs), k1, b

    def scores(self, q):
        s = defaultdict(float)
        for w in set(q):
            df = self.df.get(w)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i in self.post[w]:
                tf = self.docs[i][w]
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rank(s, k=2000):
    return sorted(s, key=lambda i: -s[i])[:k]


def rrf(*lists, k=60):
    s = defaultdict(float)
    for l in lists:
        for r, i in enumerate(l):
            s[i] += 1 / (k + r)
    return rank(s)


def evaluate(name, variants):
    os.environ["IRBENCH_KEYS"] = "1"  # scoring against the answer key
    d, run, key, qs, blocks, events = load(name)
    P = build(events, blocks)
    ids = [p["id"] for p in P]
    plain = BM([toks(p["text"]) for p in P])
    ctx = BM([toks(p["ctx"]) for p in P])
    res = defaultdict(lambda: dict(score=[], r200=[], r1000=[], r10=[]))
    for q in qs:
        atoms = S.credits(key[q["qid"]], run)
        qt = toks(q["question"])
        a, b = rank(plain.scores(qt)), rank(ctx.scores(qt))
        lists = dict(plain=a, ctx=b, rrf=rrf(a, b))
        for v, l in lists.items():
            order = [ids[i] for i in l]
            pos = {x: i for i, x in enumerate(order)}
            R = res[v]
            R["score"].append(S.score_query(key[q["qid"]], order[:400], run, atoms)["score"])
            for a_ in atoms:
                r = min((pos[x] for x in a_ if x in pos), default=99999)
                R["r10"].append(r < 10); R["r200"].append(r < 200); R["r1000"].append(r < 1000)
    for v, R in res.items():
        print(name, v, " ".join(f"{k}={st.mean(x):.3f}" for k, x in R.items()))


if __name__ == "__main__":
  for n in RUNS:
    evaluate(n, None)
