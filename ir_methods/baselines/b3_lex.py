"""b3 lexical and trivial baselines over every piece of every run (CPU only). Idempotent: a run whose marker exists is skipped."""
import random, time
import b3_lib as B
import study_helpers

TASK = "lex"
B.write_rows_and_freeze(B.LEX_TAGS)
BM_TOK = {"b3_bm25_ws": "ws", "b3_bm25_word": "word", "b3_bm25_stem": "stem", "b3_bm25_ident": "ident"}
LB = {"b3_bm25_word_lb": "b3_bm25_word", "b3_bm25_stem_lb": "b3_bm25_stem", "b3_recent_lb": "b3_recent"}


def chrono(P, rev):
    return sorted(range(len(P)), key=lambda i: (P[i]["order"], P[i]["j"], i), reverse=rev)


for split, run in B.plan():
    if B.marker(TASK, split, run).exists():
        B.log("skip", split, run); continue
    t0 = time.time()
    qs, F, R, P = study_helpers.builder(run)
    ids = [p["id"] for p in P]
    t_build = time.time() - t0
    base = {}  # tag -> (raw sub, {qid: (cand ids, scores)}, seconds)

    for tag in ("b3_recent", "b3_oldest", "b3_long", "b3_short"):
        t = time.time()
        if tag == "b3_recent": order = chrono(P, True)
        elif tag == "b3_oldest": order = chrono(P, False)
        elif tag == "b3_long": order = sorted(range(len(P)), key=lambda i: (-P[i]["tokens"], P[i]["order"], P[i]["j"], i))
        else: order = sorted(range(len(P)), key=lambda i: (P[i]["tokens"], P[i]["order"], P[i]["j"], i))
        sub = {q["qid"]: [ids[i] for i in order[:B.RAW_N]] for q in qs}
        cands = {q["qid"]: ([ids[i] for i in order[:B.LB_N]], [-float(r) for r in range(min(B.LB_N, len(order)))]) for q in qs}
        base[tag] = (sub, cands, time.time() - t)
    for seed in (0, 1, 2):
        t = time.time()
        sub = {}
        for q in qs:
            rng = random.Random(f"{seed}:{run}:{q['qid']}")
            perm = list(range(len(P))); rng.shuffle(perm)
            sub[q["qid"]] = [ids[i] for i in perm[:B.RAW_N]]
        base[f"b3_random_s{seed}"] = (sub, None, time.time() - t)
    for tag, tk in BM_TOK.items():
        t = time.time()
        f = B.TOK[tk]
        bm = B.BM25([f(p["text"]) for p in P])
        sub, cands = {}, {}
        for q in qs:
            s = bm.scores(f(q["question"]))
            o = B.order_of(s, B.RAW_N)
            sub[q["qid"]] = [ids[i] for i in o]
            cands[q["qid"]] = ([ids[i] for i in o[:B.LB_N]], [s[i] for i in o[:B.LB_N]])
        base[tag] = (sub, cands, time.time() - t)
    for tag, (sub, cands, secs) in base.items():
        B.put(tag, split, run, sub, qs); B.timing(tag, run, secs)
    for tag, src in LB.items():
        t = time.time()
        cands = base[src][1]
        sub = {q["qid"]: study_helpers.build(F, q, R, cands[q["qid"]][0], cands[q["qid"]][1]) for q in qs}
        B.put(tag, split, run, sub, qs); B.timing(tag, run, base[src][2] + time.time() - t)
    B.atomic(B.marker(TASK, split, run), dict(secs=round(time.time() - t0, 1), builder_secs=round(t_build, 1), questions=len(qs), pieces=len(P)))
    B.log("done", split, run, len(P), "pieces", len(qs), "q", f"{time.time() - t0:.1f}s")
print("LEX_ALL_DONE")
