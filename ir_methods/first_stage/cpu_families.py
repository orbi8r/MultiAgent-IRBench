"""First stage families that need no neural encoder (no keys): lexical variants, query side, conversation structure.
Each family is a ranked list of piece ids per question (top 2000). Seeded families take seeds from the lexical base list
(pure first stage) or, with the suffix _rr, from the current reranked list (a second retrieval pass after reranking).
    python cpu_families.py  -> data/<run>/fam_cpu.json {family: {qid: [ids]}}"""
import json, os, re, math
from collections import defaultdict, Counter
from run_loader import load, RUNS, words
from contextual_pieces import toks
from bm25_tools import BM, rank, rrf
from novelty import novelty
from file_provenance import Files
import build_list

build_list.SCAN, build_list.EARLIEST = 30, True
N = 2000
QX = {"who": "git commit author committed shift", "location": "mv moved file path saved folder", "history": "first earlier originally"}
ID = re.compile(r"#?\w*\d\w*|\w+[_./-]\w[\w./-]*")


def ident(text):
    """Identifiers (paths, branch names, issue numbers) and their leading parts: fix-1044-setuptools -> fix-1044 too."""
    out = []
    for m in ID.findall(text):
        m = m.lower().lstrip("#")
        if len(m) < 3:
            continue
        out.append(m)
        parts = re.split(r"([-_./])", m)
        for k in range(3, len(parts) - 1, 2):
            out.append("".join(parts[:k]))
    return out


def clauses(q):
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+(?=in which|what|who|where|when|how|which)|\bso that\b|\bafter\b|\bbefore\b", q)
    return [p for p in parts if len(p.split()) >= 3]


def seq_rank(groups):
    """Interleave groups by order: ranked flattening that keeps the seed order."""
    out = []
    for g in groups:
        out += g
    return list(dict.fromkeys(out))[:N]


for name in RUNS:
    d, run, key, qs, blocks, events = load(name)
    P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
    ids = [p["id"] for p in P]; pos = {x: i for i, x in enumerate(ids)}
    tc, tp = [toks(p["ctx"]) for p in P], [toks(p["text"]) for p in P]
    traw = [words(p["text"]) + ident(p["text"]) for p in P]
    bc, bp, br = BM(tc), BM(tp), BM(traw)
    bc2 = BM(tc, k1=2.0, b=0.3)
    nov = novelty(P)
    keep = [i for i, p in enumerate(P) if nov[p["id"]] > 0.2]
    nc, np_ = BM([tc[i] for i in keep]), BM([tp[i] for i in keep])
    ev = {e["id"]: e for e in events}
    conv_seq = defaultdict(list)
    for e in events:
        conv_seq[e["conv"]].append(e["id"])
    cpos = {eid: (e["conv"], k) for c, l in conv_seq.items() for k, eid in enumerate(l) for e in [ev[eid]]}
    by_event = defaultdict(list)
    for i, p in enumerate(P):
        by_event[p["event"]].append(i)
    F = Files(events, P); F.by_event = {k: [ids[i] for i in v] for k, v in by_event.items()}
    path_events = defaultdict(list)
    for eid, ps in F.paths.items():
        for pth in ps:
            path_events[pth].append(eid)
    by_as = defaultdict(list)
    for i, p in enumerate(P):
        by_as[(p["agent"], p["shift"])].append(i)
    # first write: earliest piece holding each 8 gram
    first = {}
    for i, p in enumerate(P):
        w = re.findall(r"[a-z0-9]+", p["text"].lower())
        for k in range(0, max(len(w) - 7, 0), 4):
            first.setdefault(hash(" ".join(w[k:k + 8])), i)
    gp = f"data/{name}/gpu_ftsyn.json"  # absent until a reader has scored the run; the _rr families then wait
    G = json.load(open(gp)) if os.path.exists(gp) else None
    out = defaultdict(dict)
    for q in qs:
        qid, qt = q["qid"], toks(q["question"])
        lc, lp = rank(bc.scores(qt), N), rank(bp.scores(qt), N)
        base = rrf(lc, lp)
        L = {"bm_ctx": lc, "bm_plain": lp, "base": base}
        L["bm_raw"] = rank(br.scores(words(q["question"]) + ident(q["question"])), N)
        L["bm_param"] = rrf(rank(bc2.scores(qt), N), lp)
        L["bm_nov"] = [keep[i] for i in rrf(rank(nc.scores(qt), N), rank(np_.scores(qt), N))]
        # RM3 style pseudo relevance feedback from the top 10 of base
        tf = Counter()
        for i in base[:10]:
            for w, c in Counter(tc[i]).items():
                tf[w] += c / len(tc[i]) * math.log(bc.n / bc.df[w])
        L["rm3"] = rrf(base, rank(bc.scores(qt + [w for w, _ in tf.most_common(25)]), N))
        cl = clauses(q["question"])
        L["qdecomp"] = rrf(base, *[rank(bc.scores(toks(c)), N) for c in cl]) if len(cl) > 1 else base
        L["qtype"] = rrf(base, rank(bc.scores(qt + toks(QX.get(q["qtype"], ""))), N))
        if G is not None:
            rr = list(dict.fromkeys(build_list.build(F, q, G["cands"][qid], G["scores"]["ft"][qid], run["text"])))
            rr = [pos[x] for x in rr if x in pos]
        for sfx, seeds in (("", base[:10]),) + ((("_rr", rr[:5]),) if G is not None else ()):
            # conversation neighbours: same event, then +-1..3 events in the same conversation
            nb = []
            for dist in range(4):
                for i in seeds:
                    c, k = cpos[P[i]["event"]]
                    for kk in {k - dist, k + dist}:
                        if 0 <= kk < len(conv_seq[c]):
                            nb += by_event[conv_seq[c][kk]]
            L["nbr" + sfx] = seq_rank([nb])
            # identifier links: rare identifiers of the seeds as a second query
            idf = Counter()
            for i in seeds:
                for t in set(ident(P[i]["text"])):
                    if br.df.get(t, 0) <= 30:
                        idf[t] += 1
            L["idlink" + sfx] = rank(br.scores([t for t, _ in idf.most_common(15)]), N) if idf else []
            # file graph: every event touching a file the seeds touched (reads, writes, moves), in time order
            fe = []
            for i in seeds:
                for pth in F.files_of(P[i]["event"]):
                    fe += path_events.get(pth, [])
            L["filegraph" + sfx] = seq_rank([by_event[e] for e in dict.fromkeys(fe)])
            # author and shift of the seeds: their pieces ranked by the question
            sc = bc.scores(qt)
            grp = []
            for i in seeds[:3]:
                grp.append(sorted(by_as[(P[i]["agent"], P[i]["shift"])], key=lambda j: -sc.get(j, 0)))
            L["authshift" + sfx] = rrf(*grp) if grp else []
            # first write of the seeds' text
            fw = []
            for i in seeds:
                w = re.findall(r"[a-z0-9]+", P[i]["text"].lower())
                fw += [first[h] for h in (hash(" ".join(w[k:k + 8])) for k in range(0, max(len(w) - 7, 0), 4)) if h in first]
            L["firstwrite" + sfx] = seq_rank([sorted(set(fw), key=fw.index)])
        L["asker"] = [i for i in rank(bc.scores(qt), 10 ** 6) if P[i]["agent"] == q["asker"]][:N]
        for k, v in L.items():
            out[k][qid] = [ids[i] for i in v]
    json.dump(out, open(f"data/{name}/fam_cpu.json", "w"))
    print(name, "families", len(out), flush=True)
