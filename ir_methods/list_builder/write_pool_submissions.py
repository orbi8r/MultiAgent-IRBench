"""Keyless submission writer for the frozen rows (list builder SCAN 30, EARLIEST on). Writes out/subs/<tag>/<run>.json.
    python write_pool_submissions.py pub      ftpub over fuse0/base top K, early exit grid, expansion from the reader's top 5 (pubsub1/2.json)
    python write_pool_submissions.py stu      student alone on fuse0 top 50/100 and the student -> ftpub cascades (data/<run>/rstul*sub1.json)
    python write_pool_submissions.py fam ADAPTER   each family's top 200 scored by ADAPTER (data/<run>/fam5_<ADAPTER>.json) -> tag fam_<family>_<ADAPTER>"""
import json, os, sys
from collections import defaultdict
from run_loader import RUNS, load
from file_provenance import Files
import build_list

build_list.SCAN, build_list.EARLIEST = 30, True
MODE = sys.argv[1]


def put(tag, name, sub):
    os.makedirs(f"out/subs/{tag}", exist_ok=True)
    json.dump(sub, open(f"out/subs/{tag}/{name}.json", "w"))


def builder(name):
    d, run, key, qs, blocks, events = load(name)
    P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
    F = Files(events, P); F.by_event = defaultdict(list); F.tokens = run["tokens"]
    for p in P:
        F.by_event[p["event"]].append(p["id"])
    return qs, F, run, P


def scores(name, tag, rnds=(1, 2)):
    E = {}
    for r in rnds:
        try:
            for qid, d in json.load(open(f"data/{name}/{tag}sub{r}.json")).items():
                E.setdefault(qid, {}).update(d)
        except FileNotFoundError:
            pass
    return E


if MODE == "pub":
    from arch1 import Run
    from arch2 import expand
    from bm25_tools import rrf
    for name in RUNS:
        r = Run(name); qs, F, run = r.qs, r.F, r.run
        F.by_event = defaultdict(list); F.tokens = run["tokens"]
        for p in r.P:
            F.by_event[p["event"]].append(p["id"])
        E = scores(name, "pub")
        T = defaultdict(dict)
        for q in qs:
            qid = q["qid"]; s = E[qid]
            L, sc = r.pass1(q)
            firsts = {"base": rrf(L["bm_ctx"], L["bm_plain"]), "fuse0": rrf(*L.values())}

            def sub(c):
                c = [r.ids[i] for i in c]
                return build_list.build(F, q, c, [s[x] for x in c], run["text"])
            f0 = firsts["fuse0"]
            for tau in (-0.01, -0.05, -0.2, -0.5, -1.0, -2.0):
                c = f0[:20] if max(s[r.ids[i]] for i in f0[:20]) >= tau else f0[:100]
                T["ftpub_exit_tau_" + ("m%s" % str(abs(tau)).replace(".", "p"))][qid] = sub(c)
            for fn, lst in firsts.items():
                for K in (20, 30, 50, 100, 200):
                    c = lst[:K]
                    if fn == "fuse0" or K == 20:
                        if fn == "fuse0":
                            T[f"ftpub_fuse0_k{K}"][qid] = sub(c)
                        else:
                            T[f"ftpub_base_k{K}"][qid] = sub(c)
                    if (fn == "fuse0") or K == 20:
                        order = sorted(c, key=lambda i: -s[r.ids[i]])
                        Lx = expand(r, q, order[:5], sc)
                        for M in (5, 10, 25):
                            cs = set(c)
                            add = list(dict.fromkeys(i for l in Lx.values() for i in l[:M] if i not in cs))
                            T[f"ftpub_exp_{fn}_k{K}_m{M}"][qid] = sub(c + add)
        for tag, sub_ in T.items():
            put(tag, name, sub_)
        print(name, len(T), "tags", flush=True)
elif MODE == "stu":
    for name in RUNS:
        qs, F, run, P = builder(name)
        L = json.load(open(f"data/{name}/f1.json"))["fuse0"]
        TT = scores(name, "pub")
        for stu in ("rstul6", "rstul12"):
            SS = scores(name, stu, (1,))
            for K1 in (50, 100):
                for N in (0, 5, 10, 20, 30):
                    if (stu, K1, N) not in (("rstul6", 50, 0), ("rstul12", 100, 0), ("rstul6", 50, 20), ("rstul6", 50, 30),
                                            ("rstul12", 100, 5), ("rstul12", 100, 10), ("rstul12", 100, 20), ("rstul12", 100, 30)) and N:
                        continue
                    sub = {}
                    for q in qs:
                        qid = q["qid"]
                        c = L[qid][:K1]
                        order = sorted(c, key=lambda x: -SS[qid][x])
                        if N == 0:
                            sc = [SS[qid][x] for x in order]
                        else:
                            lo = min(TT[qid][x] for x in order[:N]) - 1
                            sc = [TT[qid][x] if i < N else lo - i * 1e-3 for i, x in enumerate(order)]
                        sub[qid] = build_list.build(F, q, order, sc, run["text"])
                    tag = f"{stu}_fuse0_k{K1}" if N == 0 else f"cas_{stu}_k{K1}_n{N}"
                    put(tag, name, sub)
        print(name, "stu done", flush=True)
else:  # fam
    ad = sys.argv[2]
    for name in RUNS:
        qs, F, run, P = builder(name)
        fams = json.load(open(f"data/{name}/fam5_lists.json"))   # {family: {qid: [ids top 200]}}
        E = json.load(open(f"data/{name}/fam5_{ad}.json"))       # {qid: {id: score}}
        T = defaultdict(dict)
        for fam, Lq in fams.items():
            for q in qs:
                qid = q["qid"]; c = Lq[qid][:200]
                T[f"fam_{fam}_{ad}"][qid] = build_list.build(F, q, c, [E[qid][x] for x in c], run["text"])
        for tag, sub_ in T.items():
            put(tag, name, sub_)
        print(name, len(T), "tags", flush=True)
