"""The designed method's list builder: reranked candidates for every question, and for location questions the
provenance solver in front. Scores each run with the public scorer, per qtype too, and can write submissions.
    python build_list.py TAG MODEL [--write OUTDIR]"""
import json, os, sys, statistics as st
from collections import defaultdict

from file_provenance import Files

M_CONTENT = 5  # distinct files followed on location questions, fixed before any evaluation
EARLIEST = __import__("os").environ.get("EARLIEST", "0") == "1"
SCAN = int(__import__("os").environ.get("LOC_SCAN", 5))  # reranked pieces scanned for those files (5 is the original default)


def norm(t):
    return " ".join(t.split())


def ranked(cands, sc):
    return [x for _, x in sorted(zip(sc, cands), key=lambda t: -t[0])]


def dedup(lst, text, earliest=None):
    """Drop exact text repeats. With `earliest` (norm text -> earliest piece holding it in the whole record), the slot of
    the best ranked copy goes to the earliest copy, the original statement, which is what who and history credit."""
    seen, out = set(), []
    for x in lst:
        t = norm(text[x])
        if t not in seen:
            seen.add(t); out.append(earliest.get(t, x) if earliest else x)
    return out


def location_list(F, order, M=M_CONTENT, per=2):
    out, seen = [], set()
    for x in order[:SCAN]:
        if len(seen) >= M:
            break
        ev = x.split("/")[0]
        for p in F.files_of(ev)[:2]:
            fin, via = F.final(p, ev)
            if fin in seen:
                continue
            seen.add(fin)
            out += F.statements(fin, per) + [y for v in via[-1:] for y in F.by_event.get(v, [])][:1]
    return out + order


BUDGETS = (125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000)
UTIL_A = float(__import__("os").environ.get("UTIL_A", 0))  # 0 = off; else temperature on the reranker's log odds
UTIL_K = 6


def utility_order(order, score, tokens, a=UTIL_A, k=UTIL_K):
    """Reorder the top k to maximise the expected budgeted score: p_i from the reranker, a piece costs its tokens + 16,
    a budget counts if some piece read within it states the answer, budgets raised to the cheapest top piece."""
    import itertools, math
    top = order[:k]
    p, last = {}, max(score.values())
    for x in top:
        last = score.get(x, last)  # an earliest copy inherits the slot's score
        lp = min(last, -1e-6)
        lo = lp - math.log(-math.expm1(lp))
        p[x] = 1 / (1 + math.exp(-a * lo))
    F = min(tokens[x] + 16 for x in top)
    bud = [max(b, F) for b in BUDGETS]

    def value(perm):
        t, v = 0, 0.0
        reads = []
        for x in perm:
            t += tokens[x] + 16
            reads.append((t, p[x]))
        for b in bud:
            m = 1.0
            for tt, pp in reads:
                if tt <= b:
                    m *= 1 - pp
            v += 1 - m
        return v

    best = max(itertools.permutations(top), key=value)
    return list(best) + order[k:]


def build(F, q, cands, sc, text):
    if EARLIEST and not hasattr(F, "earliest"):
        F.earliest = {}
        for x in sorted(text, key=lambda x: (int(x.split("/")[0].lstrip("n") or 0), x)):
            F.earliest.setdefault(norm(text[x]), x)
    order = dedup(ranked(cands, sc), text, F.earliest if EARLIEST and q["qtype"] in ("who", "history") else None)
    if UTIL_A and q["qtype"] != "location":
        order = utility_order(order, dict(zip(cands, sc)), F.tokens)
    if q["qtype"] == "location":
        order = location_list(F, order)
    return list(dict.fromkeys(order + cands))[:400]


def main():
    from run_loader import load, S, RUNS
    tag, model = sys.argv[1], sys.argv[2]
    os.environ["IRBENCH_KEYS"] = "1"  # this entry point scores the built lists against the key
    outdir = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
    allq = []
    for name in RUNS:
        try:
            G = json.load(open(f"data/{name}/gpu_{tag}.json"))
        except FileNotFoundError:
            continue
        d, run, key, qs, blocks, events = load(name)
        P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
        F = Files(events, P)
        F.by_event = defaultdict(list)
        for p in P:
            F.by_event[p["event"]].append(p["id"])
        sc = G["scores"][model]
        sub, by = {}, defaultdict(list)
        for q in qs:
            sub[q["qid"]] = build(F, q, G["cands"][q["qid"]], sc[q["qid"]], run["text"])
            s = S.score_query(key[q["qid"]], sub[q["qid"]], run)["score"]
            by[q["qtype"]].append(s); allq.append(s)
        tot = st.mean(v for vs in by.values() for v in vs)
        print(f"{name} {tot:.3f} |", " ".join(f"{k}={st.mean(v):.2f}(n{len(v)})" for k, v in sorted(by.items())))
        if outdir:
            json.dump(sub, open(f"{outdir}/{name}.json", "w"))
    print(f"mean over questions {st.mean(allq):.3f} (n={len(allq)})")


if __name__ == "__main__":
    main()
