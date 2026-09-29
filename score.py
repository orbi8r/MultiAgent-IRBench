"""The benchmark scorer: a pure lookup, no model, same result every time.

    python3 score.py runs/<name> SUBMISSION.json > report.json
    python3 score.py KEY.json SUBMISSION.json --events EVENTS.jsonl [--blocks BLOCKS.jsonl] > report.json

KEY (hidden): {qid: {"nuggets": {nid: {id: value}}, "strata": {...}, "floor": x}}
SUBMISSION: {qid: [id, ...]} best first, where an id is a whole event ("n000123") or one block of it
("n000123/9f3c..."). A whole event is worth the best of its blocks and costs all its tokens; a block is worth only
its own label and costs only its own tokens. EVENTS and BLOCKS give the token counts.

For each budget the scorer keeps the submission's events that fit, takes the best value per nugget and averages
over nuggets, divided by the best any submission could reach under that budget (a budget where no answer piece
fits at all is skipped). Reported per query and overall: the mean over budgets, Full Support (every nugget found
in the cut), the result minus the run's floor (the best single query blind strategy) and the same per stratum.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

BUDGETS = (1000, 2000, 4000, 8000)
MIN_COST = 64


def cut(ranked, tokens, budget):
    out, used = [], 0
    for e in dict.fromkeys(ranked):
        t = max(tokens.get(e, MIN_COST), MIN_COST)
        if used + t <= budget:
            out.append(e)
            used += t
    return out, used


def score_query(entry, ranked, tokens):
    per_b, full = [], []
    for B in BUDGETS:
        kept, used = cut(ranked, tokens, B)
        best = [max((vals.get(e, 0.0) for e in kept), default=0.0) for vals in entry["nuggets"].values()]
        reach = sum(max((v for e, v in vals.items() if max(tokens.get(e, MIN_COST), MIN_COST) <= B), default=0.0)
                    for vals in entry["nuggets"].values())
        if not reach:
            continue
        per_b.append(sum(best) / reach)
        full.append(all(b > 0 for b in best))
    s = sum(per_b) / len(per_b) if per_b else 0.0
    return dict(score=s, above_floor=s - entry.get("floor", 0.0), full_support=sum(full) / len(full) if full else 0.0,
                by_budget=per_b)


def load_tokens(events_path, blocks_path=None):
    tokens = {}
    for p in (events_path, blocks_path):
        if p:
            for l in open(p, encoding="utf-8"):
                e = json.loads(l)
                tokens[e.get("id") or e.get("nid") or e.get("eid")] = e["tokens"]
    return tokens


def main(key_path, sub_path, events_path=None, blocks_path=None):
    key, sub = json.load(open(key_path, encoding="utf-8")), json.load(open(sub_path, encoding="utf-8"))
    if not events_path:
        sys.exit("give the run's events file with --events (and --blocks when you return blocks)")
    tokens = load_tokens(events_path, blocks_path)
    bad_type = [q for q, v in sub.items() if not isinstance(v, list)]
    if bad_type:
        sys.exit(f"every answer must be a list of ids; not a list for {bad_type[:5]}")
    unknown = sorted({x for v in sub.values() for x in v if x not in tokens})
    if unknown:
        sys.exit(f"{len(unknown)} ids are not in the events or blocks files given, for example {unknown[:5]}"
                 + ("" if blocks_path else " (pass --blocks when you return blocks)"))
    per_q = {q: score_query(entry, sub.get(q, []), tokens) for q, entry in key.items()}
    agg = lambda qs: {m: sum(per_q[q][m] for q in qs) / len(qs) for m in
                      ("score", "above_floor", "full_support")} if qs else {}
    strata = defaultdict(list)
    for q, entry in key.items():
        for k, v in dict(entry.get("strata", {}), qtype=entry.get("qtype")).items():
            if v:
                strata[f"{k}={v}"].append(q)
    report = dict(overall=agg(list(key)), strata={k: agg(v) for k, v in sorted(strata.items())},
                  missing=[q for q in key if q not in sub], per_query=per_q)
    json.dump(report, sys.stdout, indent=1)
    o = report["overall"]
    print(f"\nscore {o['score']:.3f} out of 1 | {o['above_floor']:+.3f} against a method that ignores the "
          f"question | every part found for {o['full_support']:.0%} of {len(key)} questions", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv[1:]
    ev = a[a.index("--events") + 1] if "--events" in a else None
    bl = a[a.index("--blocks") + 1] if "--blocks" in a else None
    if Path(a[0]).is_dir():  # short form: score.py runs/<name> my_submission.json
        d = Path(a[0])
        main(d / "answer_key.json", a[1], d / "events.jsonl", d / "blocks.jsonl")
    else:
        main(a[0], a[1], ev, bl)
