"""Novelty of every piece: share of its word 5 gram shingles never seen earlier in the record (time order). A reread or
requote of earlier content scores near 0, a first statement near 1. Built from the record only, no keys."""
import json, re


def shingles(text, n=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def novelty(P):
    seen, out = set(), {}
    for p in P:  # pieces are in record order
        s = shingles(p["text"])
        out[p["id"]] = (len(s - seen) / len(s)) if s else 0.0
        seen |= s
    return out


if __name__ == "__main__":
    from run_loader import load, S, RUNS
    for name in RUNS:
        d, run, key, qs, blocks, events = load(name)
        P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
        nov = novelty(P)
        tok = {p["id"]: p["tokens"] for p in P}
        for th in (0.0, 0.1, 0.2, 0.3, 0.5):
            keep = {x for x, v in nov.items() if v > th}
            lost = costlier = lost_loc = 0
            n = 0
            for q in qs:
                for a in S.credits(key[q["qid"]], run):
                    n += 1
                    inP = [x for x in a if x in tok]
                    k = [x for x in inP if x in keep]
                    if not k:
                        lost += 1; lost_loc += q["qtype"] == "location"
                    elif min(tok[x] for x in k) > 1.5 * min(tok[x] for x in inP) + 50:
                        costlier += 1
            print(f"{name} novelty>{th}: keep {len(keep)}/{len(P)} pieces ({100*sum(tok[x] for x in keep)/sum(tok.values()):.0f}% tokens), atoms lost {lost}/{n} (location {lost_loc}), costlier {costlier}")
