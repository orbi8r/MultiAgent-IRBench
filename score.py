"""The benchmark scorer (BAR, budgeted answer recall): a pure lookup, no model, same result every time.

    python score.py runs/<name> SUBMISSION.json > report.json

SUBMISSION: {qid: [id, ...]} best first; an id is a whole event ("n000123") or one block ("n000123/9f3c...").
The agent reads your list from the top. Each piece costs its tokens, at least 64; repeats count once.
For each question: at 1,000, 2,000, 4,000 and 8,000 tokens read, take the share of the answer's parts that a piece
read so far states. The question scores the mean of those four shares; the run scores the mean over questions.
A piece states a part when the key lists it as the first statement, a later copy or the raw tool output that shows
it. For who and history questions only the first statement counts, since its author and time are what was asked.

Half credit, the second number: some pieces state only part of an answer part (the key lists them under `partial`).
The main score counts them like any other piece. The half credit score counts such a part as half found until a
piece that states all of it is read, and `answered_at` there waits for full pieces.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

BUDGETS = (1000, 2000, 4000, 8000)
MIN_COST = 64
PARTIAL_WEIGHT = 0.5


def stating(entry):
    need = 1.0 if entry.get("qtype") in ("who", "history") else 0.4
    return [{x for x, v in vals.items() if v >= need} for vals in entry["nuggets"].values()]


def score_query(entry, ranked, tokens):
    """Per question: the four shares, their mean, and the tokens read until every part was stated (None if not
    within 8,000)."""
    parts = stating(entry)
    found_at, t = [None] * len(parts), 0
    for x in dict.fromkeys(ranked):
        t += max(tokens[x], MIN_COST)
        if t > BUDGETS[-1]:
            break
        for i, s in enumerate(parts):
            if found_at[i] is None and x in s:
                found_at[i] = t
    shares = [sum(f is not None and f <= b for f in found_at) / len(parts) for b in BUDGETS]
    answered = max(found_at) if all(f is not None for f in found_at) else None
    return dict(score=sum(shares) / len(shares), shares=shares, answered_at=answered)


def score_query_half(entry, ranked, tokens, weight=PARTIAL_WEIGHT):
    """score_query with the half credit rule: a part reached only by its partial pieces counts `weight`."""
    pmap = entry.get("partial") or {}
    parts = [(s, set(pmap.get(nid, []))) for s, nid in zip(stating(entry), entry["nuggets"])]
    full_at, half_at, t = [None] * len(parts), [None] * len(parts), 0
    for x in dict.fromkeys(ranked):
        t += max(tokens[x], MIN_COST)
        if t > BUDGETS[-1]:
            break
        for i, (s, pset) in enumerate(parts):
            if x in s:
                if x in pset:
                    if half_at[i] is None:
                        half_at[i] = t
                elif full_at[i] is None:
                    full_at[i] = t
    shares = []
    for b in BUDGETS:
        got = 0.0
        for f, h in zip(full_at, half_at):
            if f is not None and f <= b:
                got += 1.0
            elif h is not None and h <= b:
                got += weight
        shares.append(got / len(parts))
    answered = max(full_at) if all(f is not None for f in full_at) else None
    return dict(score=sum(shares) / len(shares), shares=shares, answered_at=answered)


def load_tokens(*paths):
    tokens = {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            e = json.loads(l)
            tokens[e["id"]] = e["tokens"]
    return tokens


def summary(rows):
    n = len(rows)
    out = dict(score=sum(r["score"] for r in rows) / n)
    for b in BUDGETS:
        out[f"answered_within_{b}"] = sum(r["answered_at"] is not None and r["answered_at"] <= b for r in rows) / n
    return out


def report_for(key, per_q):
    by_type = defaultdict(list)
    for q, e in key.items():
        by_type[e.get("qtype")].append(per_q[q])
    return dict(overall=summary(list(per_q.values())), by_qtype={k: summary(v) for k, v in sorted(by_type.items())},
                per_query=per_q)


def main(run_dir, sub_path):
    d = Path(run_dir)
    key = json.load(open(d / "answer_key.json", encoding="utf-8"))
    sub = json.load(open(sub_path, encoding="utf-8"))
    tokens = load_tokens(d / "events.jsonl", d / "blocks.jsonl")
    bad = [q for q, v in sub.items() if not isinstance(v, list)]
    if bad:
        sys.exit(f"every answer must be a list of ids; not a list for {bad[:5]}")
    unknown = sorted({x for v in sub.values() for x in v if x not in tokens})
    if unknown:
        sys.exit(f"{len(unknown)} ids are not in this run, for example {unknown[:5]}")
    plain = report_for(key, {q: score_query(e, sub.get(q, []), tokens) for q, e in key.items()})
    half = report_for(key, {q: score_query_half(e, sub.get(q, []), tokens) for q, e in key.items()})
    report = dict(overall=plain["overall"], by_qtype=plain["by_qtype"], missing=[q for q in key if q not in sub],
                  per_query=plain["per_query"], half_credit=half)
    json.dump(report, sys.stdout, indent=1)
    o, h = report["overall"], half["overall"]
    print(f"\nscore {o['score']:.3f} out of 1 ({h['score']:.3f} with half credit) | every part found within 2k tokens"
          f" for {o['answered_within_2000']:.0%}, within 8k for {o['answered_within_8000']:.0%} of {len(key)} questions",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python score.py runs/<run> <submission.json>")
    main(sys.argv[1], sys.argv[2])
