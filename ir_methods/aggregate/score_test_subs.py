"""Scores the frozen test submissions with the repo scorer (score.py, v5_abs).
Inputs: $IRBENCH_OUT/subs/<tag>/<run>.json for the five test runs, answer keys in $IRBENCH_RUNS.
Outputs: $IRBENCH_OUT/scores/<tag>.json and one printed line per tag. A tag is complete only with all 5 runs and no missing qid.
    python score_test_subs.py [tag ...]"""
import json, os, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import score as S

OUT = Path(os.environ["IRBENCH_OUT"])
RUNS = Path(os.environ.get("IRBENCH_RUNS") or Path(__file__).resolve().parents[2] / "runs")
TEST = ["on_call", "due_diligence", "executive_assistants", "help_desk", "textbook_team"]
tags = sys.argv[1:] or sorted(p.name for p in (OUT / "subs").iterdir() if p.is_dir())
for tag in tags:
    out, perq = {}, []
    for run in TEST:
        sub = OUT / "subs" / tag / f"{run}.json"
        if not sub.exists():
            out[run] = None
            continue
        try:
            rep = S.score_run(RUNS / run, json.load(open(sub, encoding="utf-8")))
        except Exception as e:
            out[run] = dict(error=str(e)[-300:])
            continue
        nq = sum(1 for _ in open(RUNS / run / "questions.jsonl"))
        out[run] = dict(score=rep["overall"]["score"], n=len(rep["per_query"]), questions=nq, missing=rep["missing"],
                        by_qtype={k: v["score"] for k, v in rep["by_qtype"].items()})
        perq += [v["score"] for v in rep["per_query"].values()]
    ok = all(out[r] and "score" in out[r] and not out[r]["missing"] and out[r]["n"] == out[r]["questions"] for r in TEST)
    res = dict(tag=tag, complete=ok, runs=out,
               run_mean=st.mean(out[r]["score"] for r in TEST) if ok else None,
               question_mean=st.mean(perq) if ok else None, n_questions=len(perq))
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "scores" / f"{tag}.json", "w"), indent=1)
    cells = " ".join(f"{r}={out[r]['score']:.3f}" if out[r] and "score" in out[r] else f"{r}=NA" for r in TEST)
    print(tag, "complete" if ok else "INCOMPLETE", cells, "mean %.3f" % res["run_mean"] if ok else "", f"n={len(perq)}")
