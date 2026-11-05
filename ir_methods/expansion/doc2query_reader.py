"""BM25 reader with doc2query expansions and the list builder. Keyless, it writes submissions.
Variants: none (no expansion), and per source concat (expansion tokens appended to the ctx tokens) or field L (bm_ctx + L bm_exp,
L 0.5 or 1.0). Sources: t5, qwen and t5 plus qwen. The synth source is left out, since it needs Qwen3 4B questions and was not run.
    python doc2query_reader.py t5 qwen   -> out/subs/d2q_<variant>/<run>.json"""
import json, sys, os
from collections import defaultdict
from run_loader import load, RUNS
from contextual_pieces import toks
from bm25_tools import BM, rank, rrf
from file_provenance import Files
import build_list

build_list.SCAN, build_list.EARLIEST = 30, True
SRC = sys.argv[1:]
for name in RUNS:
    d, run, key, qs, blocks, events = load(name)
    P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
    ids = [p["id"] for p in P]; pos = {x: i for i, x in enumerate(ids)}
    F = Files(events, P); F.by_event = defaultdict(list); F.tokens = run["tokens"]
    for p in P:
        F.by_event[p["event"]].append(p["id"])
    tc, tp = [toks(p["ctx"]) for p in P], [toks(p["text"]) for p in P]
    bc, bp = BM(tc), BM(tp)
    EX = {}
    for src in SRC:
        ex = defaultdict(list)
        for l in open(f"data/{name}/d2q_{src}.jsonl"):
            s = json.loads(l); ex[s["id"]] += toks(" ".join(s["q"]))
        EX[src] = ex
    if len(SRC) > 1:
        EX["+".join(SRC)] = {x: sum((EX[s].get(x, []) for s in SRC), []) for x in ids}
    V = {"none": (bc, None, 0)}
    for src, ex in EX.items():
        sn = src.replace("+", "plus")
        V[f"{sn}_concat"] = (BM([tc[i] + ex.get(x, []) for i, x in enumerate(ids)]), None, 0)
        be = BM([ex.get(x, []) for x in ids])
        for L in (0.5, 1.0):
            V[f"{sn}_field{str(L).replace('.', 'p')}"] = (bc, be, L)
    T = defaultdict(dict)
    for q in qs:
        qt = toks(q["question"])
        lp = rank(bp.scores(qt), 2000)
        for tag, (b1, b2, L) in V.items():
            sc = b1.scores(qt)
            if b2 is not None:
                s2 = b2.scores(qt)
                sc = {i: sc.get(i, 0) + L * s2.get(i, 0) for i in set(sc) | set(s2)}
            cands = [ids[i] for i in rrf(rank(sc, 2000), lp)[:200]]
            T[tag][q["qid"]] = build_list.build(F, q, cands, [sc.get(pos[x], 0) for x in cands], run["text"])
    for tag, sub in T.items():
        os.makedirs(f"out/subs/d2q_{tag}", exist_ok=True)
        json.dump(sub, open(f"out/subs/d2q_{tag}/{name}.json", "w"))
    print(name, len(T), "variants", flush=True)
