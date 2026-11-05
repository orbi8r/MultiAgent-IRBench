"""Dev only scoring and the dev gate of the structure aware study. Scores dev subs of a tag with the answer keys in $IRBENCH_RUNS through score.py (per question values only).
It reads the flat control from the pulled full2 scores (ctrl_scores/dev/<ctag>.json). It never opens a test key or test sub.
Gate: go only when the dev mean (3 runs pooled over questions) beats the control by at least GAIN and is not worse on 2 of 3 dev runs."""
import glob, json, os, sys
from pathlib import Path
BASE = Path(os.environ["IRBENCH_WORK"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import score as S
GAIN = 0.005
OUTROOT = Path(os.environ.get("OUTROOT", BASE / "out"))  # a smoke keeps its own out tree
DEV = os.environ.get("DEVR", "online_shop library_maintainers literature_review").split()


def dev_scores(tag):
    """{run: {qid: score}} of the dev subs of a tag (found under out/*/subs/dev/<tag>)."""
    out = {}
    for run in DEV:
        hit = sorted(glob.glob(str(OUTROOT / "*" / "subs" / "dev" / tag / f"{run}.json")))
        assert hit, f"no dev sub for {tag} {run}"
        d = Path(os.environ["IRBENCH_RUNS"]) / run
        sub = json.load(open(hit[0], encoding="utf-8"))
        r = S.score_run(d, sub)
        out[run] = {q: v["score"] for q, v in r["per_query"].items() if q in sub}
    return out


def ctrl_dev(ctag):
    j = json.load(open(BASE / "ctrl_scores" / "dev" / f"{ctag}.json"))
    return {run: {q: v[1] for q, v in j["runs"][run]["per_question"].items()} for run in DEV}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def pooled(sc, qids=None):
    return mean(v for run in DEV for q, v in sc[run].items() if qids is None or (run, q) in qids)


def gate(tag, ctag):
    ours, ctrl = dev_scores(tag), ctrl_dev(ctag)
    qids = {(run, q) for run in DEV for q in ours[run]}  # a smoke scores only the first questions
    per = {run: dict(ours=mean(ours[run].values()), ctrl=mean(ctrl[run][q] for q in ours[run])) for run in DEV}
    a, c = pooled(ours), mean(ctrl[run][q] for run, q in qids)
    worse = sum(per[r]["ours"] < per[r]["ctrl"] for r in DEV)
    go = a - c >= GAIN and len(DEV) - worse >= min(2, len(DEV))  # fewer dev runs only in a smoke
    why = f"dev mean {a:.4f} vs control {ctag} {c:.4f}, gain {a - c:+.4f} (need {GAIN}), worse on {worse} of {len(DEV)} dev runs (allowed {max(0, len(DEV) - 2)})"
    return dict(go=bool(go), why=why, tag=tag, control=ctag, dev_mean=a, control_dev_mean=c, gain=a - c, per_run=per, runs_worse=worse)
