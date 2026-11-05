"""Public training rows with structure prefixes. Same rows as the synth step that made pubrows/pub_NN.jsonl, but the
negative selection keeps piece ids so a row can be emitted with prefixed text: pos and negs become run_structure prefix + newline + ctx.
    python prefix_training_rows.py STAGE REFROWS OUTROOT "P1 P2"
STAGE = dir with pub_NN/{pieces,synth,events}.jsonl, REFROWS = dir with the original pubrows/pub_NN.jsonl, OUTROOT/<style>/pub_NN.jsonl written.
Env PUBSEL ("pub_00 pub_01") narrows the runs (smokes), OUT gets flatcheck.json.
Validation gate: the flat rebuild (no prefix) must equal the original file byte for byte. When a run differs the original texts are used and
ids are recovered by exact string match (ctx to id, ties broken by piece order), and flatcheck.json says which runs and how many ids were ambiguous."""
import hashlib, json, os, sys
from pathlib import Path
from contextual_pieces import toks
from bm25_tools import BM, rank
from run_structure import load_pub

stage, ref, outroot, styles = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split()
runs = os.environ.get("PUBSEL", " ".join(f"pub_{i:02d}" for i in range(15))).split()
norm = lambda t: " ".join(t.split()).lower()
dump = lambda r: json.dumps(r) + "\n"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def rebuild(P, synth):
    """The synth step, with ids: [(query, pos id, [neg ids])]."""
    ids = [p["id"] for p in P]
    ntext = {p["id"]: norm(p["text"]) for p in P}
    bm = BM([toks(p["ctx"]) for p in P])
    agents = sorted({p["agent"] for p in P})
    rows = []
    for s in synth:
        tt, ans = ntext[s["target"]], norm(s["answer"])
        asker = agents[int(hashlib.md5(s["target"].encode()).hexdigest(), 16) % len(agents)]
        q = f"{s['question']} (asked by {asker})"
        negs = [ids[i] for i in rank(bm.scores(toks(s["question"])), 60)
                if ids[i] != s["target"] and tt not in ntext[ids[i]] and ntext[ids[i]] not in tt and ans not in ntext[ids[i]]][:40]
        if len(negs) >= 7:
            rows.append((q, s["target"], negs))
    return rows


def recover(P, orig, synth):
    """Fallback: ids of the original rows by exact ctx string match. Returns rows and the number of ambiguous strings."""
    by = {}
    for p in P:
        by.setdefault(p["ctx"], []).append(p["id"])
    targets = {s["question"]: s["target"] for s in synth}
    amb, rows = 0, []
    for r in orig:
        c = by.get(r["pos"], [])
        amb += len(c) > 1
        q = r["query"].rsplit(" (asked by ", 1)[0]
        pos = targets[q] if targets.get(q) in c else (c[0] if c else None)
        negs = []
        for t in r["negs"]:
            c = [x for x in by.get(t, []) if x != pos and x not in negs]
            amb += len(by.get(t, [])) > 1
            negs.append(c[0] if c else None)
        rows.append((r["query"], pos, negs))
    return rows, amb


report = {}
for run in runs:
    P = jl(stage / run / "pieces.jsonl")
    ctx = {p["id"]: p["ctx"] for p in P}
    origp = ref / f"{run}.jsonl"
    flat = rebuild(P, jl(stage / run / "synth.jsonl"))
    text = "".join(dump(dict(query=q, pos=ctx[p], negs=[ctx[n] for n in ns])) for q, p, ns in flat)
    orig = origp.read_text() if origp.exists() else None
    same = text == orig
    info = dict(rows=len(flat), flat_bytes_equal=same, ref_found=orig is not None)
    if not same and orig is not None:
        ol, nl = orig.splitlines(True), text.splitlines(True)
        info.update(orig_rows=len(ol), same_row_set=sorted(ol) == sorted(nl), first_diff=next((i for i, (x, y) in enumerate(zip(ol, nl)) if x != y), None))
        flat, amb = recover(P, [json.loads(l) for l in ol], jl(stage / run / "synth.jsonl"))
        info.update(used="original texts with recovered ids", ambiguous=amb, unmatched=sum(p is None or None in ns for _, p, ns in flat))
        flat = [(q, p, ns) for q, p, ns in flat if p is not None and None not in ns]
    else:
        info["used"] = "rebuild"
    S = load_pub((str(stage / run), str(stage / run)))
    for st in styles:
        cache = {}
        pf = lambda i: cache.setdefault(i, S.prefix(i, st) + "\n" + ctx[i])
        d = outroot / st
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{run}.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for q, p, ns in flat:
                f.write(dump(dict(query=q, pos=pf(p), negs=[pf(n) for n in ns])))
        os.replace(tmp, d / f"{run}.jsonl")
    report[run] = info
    print(run, info, flush=True)
if os.environ.get("IRBENCH_OUT"):
    (Path(os.environ["IRBENCH_OUT"]) / "flatcheck.json").write_text(json.dumps(report, indent=1))
bad = [r for r, i in report.items() if not i["flat_bytes_equal"]]
print("FLAT_OK" if not bad else f"FLAT_DIFF {bad}", flush=True)
