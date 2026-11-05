"""Writes every grid cell as a submission (SUBDIR/algread_p..._a..._b..._k.../run.json). INTERNAL_SCORE=1 also prints the score tables.
An algorithmic reader (no neural model, nothing fitted on the runs) over the lexical top 200.
Score = z(BM25 ctx) + WP*proximity + WA*answer type span near the question terms + WB*brevity + WK*outcome kind.
Answer type comes from the question words (number, path, who, else none). Evaluated end to end with the same list
builder (build_list.build) and scorer (S.score_query). The weight grid is reported in sample and, as a diagnostic only,
leave one run out (LORO).
    python algorithmic_reader.py"""
import json, re, math, itertools, statistics as st
from collections import defaultdict
import numpy as np
import os
SCORE = os.environ.get("INTERNAL_SCORE") == "1"
if SCORE:
    os.environ["IRBENCH_KEYS"] = "1"  # answer keys are opened only for the diagnostic score table
from run_loader import load, S, RUNS
from contextual_pieces import toks
from bm25_tools import BM
from file_provenance import Files
import build_list

build_list.SCAN, build_list.EARLIEST = 30, True
NUMQ = re.compile(r"\b(how many|how much|how long|what (?:\w+ ){0,3}(?:price|figure|number|rate|cost|margin|percentage|value|total|count|median|mean|average|amount|score|size|share|ratio|version|threshold|cutoff|limit))", re.I)
NUM = re.compile(r"[£$€]?\d[\d,]*(?:\.\d+)?%?")
PATH = re.compile(r"\w[\w.-]*/[\w./-]+")
W = 12


def atype(q):
    t = q["question"].lower()
    if q["qtype"] == "location" or t.startswith("where"):
        return "path"
    if q["qtype"] == "who" or t.startswith("who"):
        return "who"
    if NUMQ.search(t):
        return "num"
    return None


def feats(q, p, qs_set, at):
    words = re.findall(r"\S+", p["text"])[:400]
    st_ = [set(toks(w)) for w in words]
    pos = [i for i, s in enumerate(st_) if s & qs_set]
    # proximity: shortest window holding min(3, distinct matched) distinct question terms
    prox = 0.0
    if pos:
        need = min(3, len(set().union(*(st_[i] & qs_set for i in pos))))
        best = 10 ** 6
        for a in range(len(pos)):
            seen = set()
            for b in range(a, len(pos)):
                seen |= st_[pos[b]] & qs_set
                if len(seen) >= need:
                    best = min(best, pos[b] - pos[a]); break
        prox = 1 / (1 + best / 5) if best < 10 ** 6 else 0
    span = 0.0
    if at and pos:
        near = set(j for i in pos for j in range(max(0, i - W), min(len(words), i + W + 1)))
        if at == "num":
            span = any(NUM.fullmatch(words[j].strip("*(),.:;")) for j in near)
        elif at == "path":
            span = any(PATH.search(words[j]) for j in near)
        elif at == "who":
            span = bool(re.search(r"\b(commit|author|merged|pushed)\b", p["text"], re.I))
    kind = 1.0 if (p["kind"] == "tool_result" or "write_to_file" in p["ctx"][:200] or "replace_file_content" in p["ctx"][:200]) else 0.0
    return [prox, float(span), -math.log(1 + p["tokens"]), kind]


GRID = list(itertools.product((0, 1, 2), (0, 0.5, 1), (0, 0.2, 0.5), (0, 0.5)))
T, PQ, SUB = {}, {}, {}
SUBDIR = os.environ.get("SUBDIR", "subs_algread")


def tag(g):
    return "algread_p%s_a%s_b%s_k%s" % tuple(str(x).replace(".", "d") for x in g)

for name in RUNS:
    d, run, key, qs, blocks, events = load(name)
    P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
    pid = {p["id"]: p for p in P}
    F = Files(events, P); F.by_event = defaultdict(list); F.tokens = run["tokens"]
    for p in P:
        F.by_event[p["event"]].append(p["id"])
    bc = BM([toks(p["ctx"]) for p in P]); idx = {p["id"]: i for i, p in enumerate(P)}
    base = json.load(open(f"data/{name}/fam_cpu.json"))["base"]
    for q in qs:
        cands = base[q["qid"]][:200]
        qt = toks(q["question"]); qs_set = set(qt)
        sc = bc.scores(qt); b = np.array([sc.get(idx[x], 0) for x in cands]); b = (b - b.mean()) / (b.std() + 1e-9)
        at = atype(q)
        X = np.array([feats(q, pid[x], qs_set, at) for x in cands])
        X[:, 2] = (X[:, 2] - X[:, 2].mean()) / (X[:, 2].std() + 1e-9)
        for g in GRID:
            s = b + X @ np.array(g)
            lst = build_list.build(F, q, cands, s.tolist(), run["text"])
            SUB.setdefault((name, g), {})[q["qid"]] = lst
            if SCORE:
                PQ.setdefault((name, g), []).append(S.score_query(key[q["qid"]], lst, run)["score"])
    print(name, "done", flush=True)
for (name, g), sub in SUB.items():
    os.makedirs(f"{SUBDIR}/{tag(g)}", exist_ok=True)
    json.dump(sub, open(f"{SUBDIR}/{tag(g)}/{name}.json", "w"))
if SCORE:
    for g in GRID:
        T[g] = {n: st.mean(PQ[(n, g)]) for n in RUNS}
    z = (0, 0, 0, 0)
    print("BM25 only", {n: round(v, 3) for n, v in T[z].items()}, "mean %.3f" % st.mean(T[z].values()))
    best = sorted(GRID, key=lambda g: -st.mean(T[g].values()))[:8]
    for g in best:
        print("weights", g, {n: round(v, 3) for n, v in T[g].items()}, "mean %.3f" % st.mean(T[g].values()))
    held = []
    for test in RUNS:
        g = max(GRID, key=lambda g: sum(T[g][n] for n in RUNS if n != test))
        held += PQ[(test, g)]; print("LORO", test, g, "%.3f" % T[g][test])
    print("LORO question mean %.3f" % st.mean(held))
