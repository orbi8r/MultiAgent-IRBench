"""Write frozen test submissions ($IRBENCH_OUT/subs/<row>/<run>.json) from the saved GPU outputs, through the frozen list builder
(EARLIEST=1 LOC_SCAN=30 UTIL_A=0, set in this file, not by env). Run with cwd $IRBENCH_WORK. Never reads a key except through the
scorer wrapper aggregate/score_test_subs.py afterwards.
    python write_submissions.py gpu ROW TAG MODEL                      reader scores of data/<run>/gpu_<TAG>.json[scores][MODEL]
    python write_submissions.py raw ROW TAG MODEL                      same scores, plain score order, no list builder
    python write_submissions.py first ROW TAG [K]                      first stage list as is (cands order, first K)
    python write_submissions.py ens ROW TAGA MODELA TAGB MODELB W      z(A) + W z(B) per question on the cands of file A
    python write_submissions.py lw ROW BASETAG BASEMODEL LISTTAG       listwise picks first, then the base order
    python write_submissions.py casc ROW STUDTAG STUDMODEL SLOWTAG SLOWMODEL M   student orders, slow reader rereads the top M
"""
import json, os, sys, statistics as st
from collections import defaultdict
os.environ["EARLIEST"], os.environ["LOC_SCAN"], os.environ["UTIL_A"] = "1", "30", "0"
from run_loader import load, RUNS
from file_provenance import Files
import build_list

build_list.SCAN, build_list.EARLIEST, build_list.UTIL_A = 30, True, 0
OUT = os.environ["IRBENCH_OUT"] + "/subs"


def z(v):
    m, s = st.mean(v), st.pstdev(v) or 1.0
    return [(x - m) / s for x in v]


def jl(path):
    return json.load(open(path))


def main():
    mode, row = sys.argv[1], sys.argv[2]
    a = sys.argv[3:]
    os.makedirs(f"{OUT}/{row}", exist_ok=True)
    for name in RUNS:
        d, run, key, qs, blocks, events = load(name)
        P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
        F = Files(events, P)
        F.by_event = defaultdict(list)
        for p in P:
            F.by_event[p["event"]].append(p["id"])
        sub = {}
        if mode == "gpu":
            G = jl(f"data/{name}/gpu_{a[0]}.json"); cands, sc = G["cands"], G["scores"][a[1]]
        elif mode == "first":
            G = jl(f"data/{name}/gpu_{a[0]}.json")
            K = int(a[1]) if len(a) > 1 else 200
            sub = {q["qid"]: G["cands"][q["qid"]][:K] for q in qs}
        elif mode == "raw":
            G = jl(f"data/{name}/gpu_{a[0]}.json")
            sub = {q["qid"]: [x for _, x in sorted(zip(G["scores"][a[1]][q["qid"]], G["cands"][q["qid"]]), key=lambda t: -t[0])] for q in qs}
        elif mode == "ens":
            A, B = jl(f"data/{name}/gpu_{a[0]}.json"), jl(f"data/{name}/gpu_{a[2]}.json")
            w = float(a[4]); cands = A["cands"]; sc = {}
            for q in qs:
                qid = q["qid"]
                assert B["cands"][qid] == cands[qid], "candidate lists differ"
                sc[qid] = [x + w * y for x, y in zip(z(A["scores"][a[1]][qid]), z(B["scores"][a[3]][qid]))]
        elif mode == "lw":
            G = jl(f"data/{name}/gpu_{a[0]}.json"); L = jl(f"data/{name}/list_{a[2]}.json")
            cands, sc = G["cands"], {}
            for q in qs:
                qid = q["qid"]
                picks = [L[qid]["order"][i] for i in L[qid]["picks"]]
                boost = {x: 1000 - r for r, x in enumerate(picks)}
                sc[qid] = [boost.get(x, s) for x, s in zip(cands[qid], G["scores"][a[1]][qid])]
        elif mode == "casc":
            A, B = jl(f"data/{name}/gpu_{a[0]}.json"), jl(f"data/{name}/gpu_{a[2]}.json")
            M = int(a[4]); cands, sc = {}, {}
            for q in qs:
                qid = q["qid"]
                c, sa = A["cands"][qid], A["scores"][a[1]][qid]
                slow = dict(zip(B["cands"][qid], B["scores"][a[3]][qid]))
                order = [x for _, x in sorted(zip(sa, c), key=lambda t: -t[0])]
                head = sorted([x for x in order[:M] if x in slow], key=lambda x: -slow[x])
                hs = set(head)
                new = head + [x for x in order if x not in hs]
                cands[qid] = new; sc[qid] = [len(new) - i for i in range(len(new))]
        else:
            sys.exit(__doc__)
        if mode not in ("first", "raw"):
            sub = {q["qid"]: build_list.build(F, q, cands[q["qid"]], sc[q["qid"]], run["text"]) for q in qs}
        assert all(len(v) <= 400 for v in sub.values())
        json.dump(sub, open(f"{OUT}/{row}/{name}.json", "w"))
        print(row, name, len(sub), "questions")


main()
