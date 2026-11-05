"""Scores every submission with the repo scorer.
    python score_submissions.py [--subs SUBDIR] [--out-name NAME]
Reads $IRBENCH_WORK/out/*/SUBDIR/{dev,test}/<tag>/<run>.json (SUBDIR default subs, a task folder is skipped when it is the output
folder of this call) and writes $IRBENCH_OUT/NAME/{dev,test}/<tag>.json (NAME default scores) =
{tag, split, task, complete, runs: {run: {score, n, questions, missing, error,
by_qtype: {qtype: {score, n}}, per_question: {qid: [qtype, score]}, equal_score}}}. A tag whose scores file is newer than its subs is skipped.
The baselines study calls it with --subs b3subs --out-name b3scores and the glob narrowed by TASKGLOB=*_b3_*. DEVR and TESTR name the runs per split."""
import glob, json, os, sys
from pathlib import Path
BASE, OUT = Path(os.environ["IRBENCH_WORK"]), Path(os.environ["IRBENCH_OUT"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import score as S
DEV = os.environ.get("DEVR", "online_shop library_maintainers literature_review").split()
TEST = os.environ.get("TESTR", "on_call due_diligence executive_assistants help_desk textbook_team").split()
ARGS = dict(zip(sys.argv[1::2], sys.argv[2::2]))
SUBS, NAME = ARGS.get("--subs", "subs"), ARGS.get("--out-name", "scores")
CACHE = {}


def keyrun(split, run):
    if (split, run) not in CACHE:
        d = Path(os.environ["IRBENCH_RUNS"]) / run
        CACHE[(split, run)] = (d, json.load(open(d / "answer_key.json", encoding="utf-8")), S.load_run(d))
    return CACHE[(split, run)]


def one(split, run, sub_path):
    d, key, run_ = keyrun(split, run)
    sub = json.load(open(sub_path, encoding="utf-8"))
    bad = [q for q, v in sub.items() if not isinstance(v, list) or not all(isinstance(x, str) for x in v)]
    if bad:
        return dict(error=f"not id lists for {bad[:5]}")
    unknown = sorted({x for v in sub.values() for x in v if x not in run_["tokens"]})
    if unknown:
        return dict(error=f"{len(unknown)} ids not in the run, e.g. {unknown[:3]}")
    r = S.score_run(d, sub, key=key, run=run_)
    bt = {}
    for q, e in key.items():
        bt.setdefault(e.get("qtype"), []).append(r["per_query"][q]["score"])
    return dict(score=r["overall"]["score"], n=len(r["per_query"]), questions=len(key), missing=r["missing"],
                by_qtype={k: dict(score=sum(v) / len(v), n=len(v)) for k, v in sorted(bt.items())},
                per_question={q: [key[q].get("qtype"), r["per_query"][q]["score"]] for q in key},
                equal_score=r["equal_credit"]["overall"]["score"])


for split, runs in (("dev", DEV), ("test", TEST)):
    found = {}
    for p in sorted(glob.glob(str(BASE / "out" / os.environ.get("TASKGLOB", "*") / SUBS / split / "*"))):
        tag = Path(p).name; task = Path(p).parts[-4]
        if task == OUT.name:
            continue
        if tag in found:
            print("DUPLICATE tag", tag, found[tag][0], task, flush=True)
            continue
        found[tag] = (task, Path(p))
    (OUT / NAME / split).mkdir(parents=True, exist_ok=True)
    for tag, (task, p) in sorted(found.items()):
        dst = OUT / NAME / split / f"{tag}.json"
        subs = {r: p / f"{r}.json" for r in runs if (p / f"{r}.json").exists()}
        if dst.exists() and subs and dst.stat().st_mtime > max(f.stat().st_mtime for f in subs.values()):
            continue
        res = {}
        for r in runs:
            res[r] = one(split, r, subs[r]) if r in subs else None
        ok = all(res[r] and "score" in res[r] and not res[r]["missing"] and res[r]["n"] == res[r]["questions"] for r in runs)
        json.dump(dict(tag=tag, split=split, task=task, complete=ok, runs=res), open(dst, "w"))
        cells = " ".join(f"{r}={res[r]['score']:.3f}" if res[r] and "score" in res[r] else f"{r}=NA" for r in runs)
        print(split, tag, "complete" if ok else "INCOMPLETE", cells, flush=True)
