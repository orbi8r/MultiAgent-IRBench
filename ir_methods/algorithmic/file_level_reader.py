"""Retrieves FILES, not pieces, for location questions, and writes each variant as a full submission
(SUBDIR/filelevel_<variant>/run.json). Location questions carry the variant, the others carry the BM25 reader list
builder output. Nothing is fitted and the method uses no keys. INTERNAL_SCORE=1 also opens the keys and prints the score table.
A file's document = every piece of every event whose tool call names that path (writes and reads), or writes only.
Files ranked by BM25 over those documents, fused (RRF) with the files met while scanning the reader's top pieces; the
top M=5 files go through the same provenance solver (final path after moves, cheapest statements, last move).
Readers: bm25 (ctx BM25 over the lexical top 200) and ft (the fine tuned fold scores, as a reference only).
    python file_level_reader.py"""
import json, os, re, statistics as st
from collections import defaultdict
SCORE = os.environ.get("INTERNAL_SCORE") == "1"
if SCORE:
    os.environ["IRBENCH_KEYS"] = "1"  # answer keys are opened only for the diagnostic score table
from run_loader import load, S, RUNS
from contextual_pieces import toks
from bm25_tools import BM, rank
import build_list
from file_provenance import Files

build_list.SCAN, build_list.EARLIEST = 30, True
WR = re.compile(r"write_to_file|replace_file_content|CodeContent|ReplacementContent|> ?\S+\.\w+")
R = defaultdict(list)
SUB = {}  # variant -> run -> {qid: list}; location questions carry the variant, the rest the BM25 reader list


def loc_list(F, files, order, per=2, M=5):
    out, seen = [], set()
    for p, ev in files:
        if len(seen) >= M:
            break
        fin, via = F.final(p, ev)
        if fin in seen:
            continue
        seen.add(fin)
        out += F.statements(fin, per) + [y for v in via[-1:] for y in F.by_event.get(v, [])][:1]
    return list(dict.fromkeys(out + order))[:400]


for name in RUNS:
    d, run, key, qs, blocks, events = load(name)
    P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
    ids = [p["id"] for p in P]; pos = {x: i for i, x in enumerate(ids)}
    F = Files(events, P); F.by_event = defaultdict(list); F.tokens = run["tokens"]
    for p in P:
        F.by_event[p["event"]].append(p["id"])
    ev_of = {e["id"]: e for e in events}
    pe = defaultdict(list)
    for eid, ps in F.paths.items():
        for pth in ps:
            pe[pth].append(eid)
    paths = sorted(pe)
    docs_all, docs_wr = [], []
    for pth in paths:
        a, w = [], []
        for eid in pe[pth]:
            tt = []
            for x in F.by_event.get(eid, []) + [r for r, c in F.call_of.items() if c["id"] == eid for r in [r] for x2 in [0]][:0]:
                tt += toks(P[pos[x]]["ctx"])
            # results of this call
            a += tt
            if WR.search(ev_of[eid]["text"][:3000]):
                w += tt
        docs_all.append(a); docs_wr.append(w)
    # add tool result pieces to their call's file docs
    res_of = defaultdict(list)
    for rid, c in F.call_of.items():
        res_of[c["id"]].append(rid)
    for k, pth in enumerate(paths):
        for eid in pe[pth]:
            for rid in res_of.get(eid, []):
                docs_all[k] += sum((toks(P[pos[x]]["ctx"]) for x in F.by_event.get(rid, [])), [])
    BA, BW = BM(docs_all), BM(docs_wr)
    bc = BM([toks(p["ctx"]) for p in P])
    base = json.load(open(f"data/{name}/fam_cpu.json"))["base"]
    gp = f"data/{name}/gpu_ftsyn.json"
    G = json.load(open(gp)) if os.path.exists(gp) else None
    for q in qs:
        qt = toks(q["question"]); atoms = S.credits(key[q["qid"]], run) if SCORE else None
        sc = bc.scores(qt)
        cb = base[q["qid"]][:200]
        if q["qtype"] != "location":  # every non location question gets the BM25 reader list builder output
            SUB.setdefault("bm25_solver", {}).setdefault(name, {})[q["qid"]] = build_list.build(F, q, cb, [sc.get(pos[x], 0) for x in cb], run["text"])
            continue
        readers = {"bm25": (cb, None)}
        if G is not None:
            readers["ft"] = (G["cands"][q["qid"]], G["scores"]["ft"][q["qid"]])
        for rn, (cands, s) in readers.items():
            if s is None:
                s = [sc.get(pos[x], 0) for x in cands]
            order = build_list.dedup(build_list.ranked(cands, s), run["text"])
            if SCORE:
                R[f"{rn}_solver"].append(S.score_query(key[q["qid"]], build_list.location_list(F, order) + order, run, credit=atoms)["score"])
            SUB.setdefault(f"{rn}_solver", {}).setdefault(name, {})[q["qid"]] = list(dict.fromkeys(build_list.location_list(F, order) + order))[:400]
            scan = []
            for x in order[:30]:
                ev = x.split("/")[0]
                scan += [(p, ev) for p in F.files_of(ev)[:2]]
            scan = list(dict.fromkeys(scan))
            for dn, B in (("all", BA), ("wr", BW)):
                fr = [(paths[i], pe[paths[i]][0]) for i in rank(B.scores(qt), 50)]
                fused = defaultdict(float)
                for r, f in enumerate(scan):
                    fused[f[0]] += 1 / (60 + r)
                for r, f in enumerate(fr):
                    fused[f[0]] += 1 / (60 + r)
                first = {}
                for f in scan + fr:
                    first.setdefault(f[0], f[1])
                files = [(p, first[p]) for p in sorted(fused, key=lambda p: -fused[p])]
                for kind, fl in (("only", fr), ("fused", files)):
                    lst = loc_list(F, fl, order)
                    if SCORE:
                        R[f"{rn}_files_{dn}_{kind}"].append(S.score_query(key[q["qid"]], lst, run, credit=atoms)["score"])
                    SUB.setdefault(f"{rn}_files_{dn}_{kind}", {}).setdefault(name, {})[q["qid"]] = lst
    print(name, flush=True)
SUBDIR = os.environ.get("SUBDIR", "subs_filelevel")
for var, byrun in SUB.items():
    for name, sub in byrun.items():
        # a non solver variant needs the BM25 list for non location questions
        if var != "bm25_solver":
            for qid, l in SUB["bm25_solver"][name].items():
                sub.setdefault(qid, l) if qid not in sub else None
        os.makedirs(f"{SUBDIR}/filelevel_{var}", exist_ok=True)
        json.dump(sub, open(f"{SUBDIR}/filelevel_{var}/{name}.json", "w"))
for k in sorted(R):
    print(f"{k:28s} {st.mean(R[k]):.3f} n={len(R[k])}")
