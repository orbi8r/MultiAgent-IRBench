"""The benchmark scorer (BAR, budgeted answer recall): a pure lookup, no model, same result every time.

    python score.py runs/<name> SUBMISSION.json > report.json

SUBMISSION: {qid: [id, ...]} best first; an id is a whole event ("n000123") or one block ("n000123/9f3c...").
The agent reads your list from the top. Each piece costs its tokens plus OVERHEAD = 16 (the id and separators around it);
repeats are free. Reading stops at the piece that crosses the largest budget.

Main score. Each answer is split into atoms, the separate facts it needs (the key's `atoms`, one {id: mark} per atom).
A piece earns credit 1 for every atom it states and 0 for the rest. A piece states an atom when
  it is listed in that atom (the first statement or any later copy the key judged to state it), or
  its text, whitespace normalised, equals the text of a listed piece, or holds it whole when that text is at least 5 words.
Who and history questions ask for an author and a time, so there a piece other than the first statement states the atom
only if what the reader sees of it (its text, its event's agent and shift) carries the first statement's agent and shift,
or it holds the first statement's text whole and that text is the answer: always on history (the content asked about),
on who when the text itself names an agent and a shift.
The ideal cost F of a question is the cheapest set of pieces that states every atom. Budgets are 125, 250, 500, 1,000,
2,000, 4,000, 8,000, 16,000 and 32,000 tokens, each raised to at least F. At each budget a question's share is the
fraction of its atoms stated by a piece read within it, and the question scores the mean of the nine shares: the area
under its answer curve on a log axis of tokens read. Reading stops past the last budget.
A list that states every atom within F scores 1, and nothing can do better. The run scores the mean over questions.
No constant depends on the run or on how many questions it has.

Equal credit, the second number, is the released rule unchanged: each piece costs its tokens, at least 64, budgets
1,000 to 8,000, a piece that states any atom finds the whole answer, copies count like the first statement, only the
first statement counts for who and history.
"""
import bisect
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BUDGETS = (125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000)
OVERHEAD = 16
MIN_WORDS = 5
PROVENANCE = ("who", "history")
FIRST_ONLY = PROVENANCE
EQUAL_BUDGETS = (1000, 2000, 4000, 8000)
EQUAL_MIN_COST = 64


def cost(x, tokens):
    return max(tokens[x], 1) + OVERHEAD


def read_order(ranked, tokens, last=float("inf")):
    """(piece, tokens read once it is read) for every piece read within `last` tokens."""
    t = 0
    for x in dict.fromkeys(ranked):
        t += cost(x, tokens)
        if t > last:
            return
        yield x, t


def shares(found, budgets):
    """found: per atom or part, a list of (tokens read, credit). -> mean share over the budgets and the shares."""
    s = [sum(max((c for t, c in f if t <= b), default=0.0) for f in found) / len(found) for b in budgets]
    return sum(s) / len(s), s


def norm(t):
    return " ".join(t.split())


def _index(run):
    """Normalised text of every piece: exact lookup and one joined string for containment search. Built once per run."""
    if "_idx" not in run:
        ids = list(run["text"])
        texts = [norm(run["text"][x]) for x in ids]
        exact = defaultdict(list)
        for x, t in zip(ids, texts):
            exact[t].append(x)
        starts, pos = [], 0
        for t in texts:
            starts.append(pos)
            pos += len(t) + 1
        run["_idx"] = dict(ids=ids, exact=exact, starts=starts, blob="\x00".join(texts), memo={})
    return run["_idx"]


def same_text(run, x):
    """Ids whose normalised text equals the text of x, or holds it whole when that text has at least MIN_WORDS words."""
    ix = _index(run)
    if x in ix["memo"]:
        return ix["memo"][x]
    t = norm(run["text"][x])
    out = set(ix["exact"].get(t, ()))
    if len(re.findall(r"[a-z0-9]+", t.lower())) >= MIN_WORDS:
        blob, starts, ids = ix["blob"], ix["starts"], ix["ids"]
        i = blob.find(t)
        while i >= 0:
            out.add(ids[bisect.bisect_right(starts, i) - 1])
            i = blob.find(t, i + 1)
    ix["memo"][x] = out
    return out


def carries(run, x, agent, shift):
    """True when the reader of x sees the asked agent and shift, in its text or in its event's own agent and shift."""
    e, t = x.split("/")[0], run["text"][x]
    who = run["agent"][e] == agent or re.search(r"(?<![\w-])" + re.escape(agent) + r"(?![\w-])", t)
    when = run["shift"][e] == shift or re.search(r"(?i)\bshift\W{0,3}%d\b" % shift, t)
    return bool(who and when)


def credits(entry, run):
    """Per atom {id: 1.0} for every piece that states it."""
    out = []
    for a in entry["atoms"]:
        firsts = [x for x, m in a.items() if m >= 1.0]
        ids = set(a).union(*(same_text(run, x) for x in a))
        if entry.get("qtype") in PROVENANCE:
            src = ([x for x in firsts if "/" in x] or firsts)[0]
            e, t0 = src.split("/")[0], norm(run["text"][src])
            # a copy of the first statement's text shows what that text shows: the content (history), or the author
            # and shift when the text itself names them (who); otherwise the copy must show the first statement's own
            in_text = entry["qtype"] == "history" or (re.search(r"(?i)\bshift\W{0,3}\d+\b", t0) and any(
                re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", t0) for a in set(run["agent"].values())))
            ids = {x for x in ids if x in firsts or carries(run, x, run["agent"][e], run["shift"][e])
                   or (in_text and t0 in norm(run["text"][x]))}
        out.append({x: 1.0 for x in ids})
    return out


def ideal(atoms, tokens):
    """(F, pieces): the cheapest set of pieces that states every atom (exact, over subsets of atoms)."""
    n = len(atoms)
    mask = defaultdict(int)
    for i, a in enumerate(atoms):
        for x in a:
            mask[x] |= 1 << i
    cheapest = {}
    for x, k in mask.items():
        if k not in cheapest or (cost(x, tokens), x) < (cost(cheapest[k], tokens), cheapest[k]):
            cheapest[k] = x
    best = {0: (0, ())}
    for m in range(1 << n):
        if m not in best:
            continue
        c0, p0 = best[m]
        for k, x in cheapest.items():
            m2 = m | k
            if m2 != m:
                c = c0 + cost(x, tokens)
                if m2 not in best or (c, p0 + (x,)) < best[m2]:
                    best[m2] = (c, p0 + (x,))
    c, p = best[(1 << n) - 1]
    return c, sorted(p, key=lambda x: (cost(x, tokens), x))


def score_query(entry, ranked, run, credit=None):
    """Main score of one question."""
    atoms = credit or credits(entry, run)
    F = ideal(atoms, run["tokens"])[0]
    found = [[] for _ in atoms]
    for x, t in read_order(ranked, run["tokens"], max(F, BUDGETS[-1])):
        for f, a in zip(found, atoms):
            if a.get(x):
                f.append((t, a[x]))
    score, s = shares(found, [max(F, b) for b in BUDGETS])
    full = [min((t for t, c in f), default=None) for f in found]
    return dict(score=score, shares=s, ideal_cost=F, answered_at=max(full) if all(full) else None)


def read_order_equal(ranked, tokens, last):
    t = 0
    for x in dict.fromkeys(ranked):
        t += max(tokens[x], EQUAL_MIN_COST)
        if t > last:
            return
        yield x, t


def score_query_equal(entry, ranked, run):
    """The released equal credit rule: any listed piece finds the part (only the first statement for who and history)."""
    need = 1.0 if entry.get("qtype") in FIRST_ONLY else 0.4
    parts = [{x for x, v in vals.items() if v >= need} for vals in entry["nuggets"].values()]
    found = [[] for _ in parts]
    for x, t in read_order_equal(ranked, run["tokens"], EQUAL_BUDGETS[-1]):
        for f, p in zip(found, parts):
            if x in p:
                f.append((t, 1.0))
    first = [min((t for t, c in f), default=None) for f in found]
    score, s = shares(found, EQUAL_BUDGETS)
    return dict(score=score, shares=s, answered_at=max(first) if all(first) else None)


def load_run(d):
    run = dict(tokens={}, text={}, conv={}, agent={}, shift={})
    for p in ("events.jsonl", "blocks.jsonl"):
        for l in open(Path(d) / p, encoding="utf-8"):
            e = json.loads(l)
            run["tokens"][e["id"]] = e["tokens"]
            run["text"][e["id"]] = e["text"]
            for k in ("conv", "agent", "shift"):
                if k in e:
                    run[k][e["id"]] = e[k]
    return run


def summary(rows, main=True):
    n = len(rows)
    out = dict(score=sum(r["score"] for r in rows) / n)
    if main:
        for b in BUDGETS:
            out[f"answered_within_{b}"] = sum(r["answered_at"] is not None and r["answered_at"] <= max(b, r["ideal_cost"]) for r in rows) / n
    else:
        for b in EQUAL_BUDGETS:
            out[f"answered_within_{b}"] = sum(r["answered_at"] is not None and r["answered_at"] <= b for r in rows) / n
    return out


def report_for(key, per_q, main=True):
    by_type = defaultdict(list)
    for q, e in key.items():
        by_type[e.get("qtype")].append(per_q[q])
    return dict(overall=summary(list(per_q.values()), main), by_qtype={k: summary(v, main) for k, v in sorted(by_type.items())},
                per_query=per_q)


def score_run(d, sub, key=None, run=None):
    d = Path(d)
    key = key or json.load(open(d / "answer_key.json", encoding="utf-8"))
    run = run or load_run(d)
    main = report_for(key, {q: score_query(e, sub.get(q, []), run) for q, e in key.items()})
    equal = report_for(key, {q: score_query_equal(e, sub.get(q, []), run) for q, e in key.items()}, main=False)
    return dict(overall=main["overall"], by_qtype=main["by_qtype"], missing=[q for q in key if q not in sub],
                per_query=main["per_query"], equal_credit=equal)


def main(run_dir, sub_path):
    d = Path(run_dir)
    sub = json.load(open(sub_path, encoding="utf-8"))
    run = load_run(d)
    bad = [q for q, v in sub.items() if not isinstance(v, list) or not all(isinstance(x, str) for x in v)]
    if bad:
        sys.exit(f"every answer must be a list of id strings; not one for {bad[:5]}")
    unknown = sorted({x for v in sub.values() for x in v if x not in run["tokens"]})
    if unknown:
        sys.exit(f"{len(unknown)} ids are not in this run, for example {unknown[:5]}")
    report = score_run(d, sub, run=run)
    json.dump(report, sys.stdout, indent=1)
    o, e = report["overall"], report["equal_credit"]["overall"]
    print(f"\nscore {o['score']:.3f} out of 1 (equal credit {e['score']:.3f}) | every atom found within 2k tokens"
          f" for {o['answered_within_2000']:.0%}, within 32k for {o['answered_within_32000']:.0%}"
          f" of {len(report['per_query'])} questions", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python score.py runs/<run> <submission.json>")
    main(sys.argv[1], sys.argv[2])
