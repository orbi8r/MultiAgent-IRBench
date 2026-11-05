"""Builds results_full3b.csv, the baseline table, with the column layout of results/results_test5.csv.
Reads $IRBENCH_WORK/out/*_b3_*/{b3rows,b3timing,b3scores}: rows and timing written by the lex and dense steps, scores written by the score step.
Writes $IRBENCH_OUT/results_full3b.csv. No compute beyond means. Exit code 0 when all 22 tags are complete on all 8 runs, 1 otherwise
(the csv is still written with what exists). Prints one status line. Run it through recipes/full3b/99_aggregate.sh, which sources env.sh."""
import csv, glob, json, os, statistics as st, sys
from pathlib import Path
import b3_lib as B

O = Path(os.environ["IRBENCH_WORK"]) / "out"
D = Path(os.environ["IRBENCH_OUT"])
DEV = ["online_shop", "library_maintainers", "literature_review"]
TEST = ["on_call", "due_diligence", "executive_assistants", "help_desk", "textbook_team"]
COLS = ["tag", "section", "method", "doc_kind", "test_clean", "frozen_setting", "dev_online", "dev_library", "dev_literature", "dev_mean",
        "test_on_call", "test_due_diligence", "test_executive_assistants", "test_help_desk", "test_textbook_team", "test_mean", "test_minus_dev",
        "dev_time", "test_time", "completeness", "clean_reason", "dev_note", "notes"]
rows, timing = {}, {}
for f in glob.glob(str(O / "*_b3_*" / "b3rows" / "*.json")):
    r = json.load(open(f)); rows[r["tag"]] = r
for f in glob.glob(str(O / "*_b3_*" / "b3timing" / "*.json")):
    timing[Path(f).stem] = json.load(open(f))
scores = {s: {Path(p).stem: json.load(open(p)) for p in glob.glob(str(O / "*_b3_*" / "b3scores" / s / "*.json"))} for s in ("dev", "test")}


def f4(x):
    return "" if x is None else f"{x:.4f}"


out, ncomplete = [], 0
for tag in B.TAGS:
    r = rows.get(tag, dict(section=B.SECTION, method=B.TAGS[tag][0], doc_kind="FIXED", frozen_setting=B.TAGS[tag][1], clean_reason=B.CLEAN_REASON, dev_note="", notes=""))
    sd, stt = scores["dev"].get(tag), scores["test"].get(tag)
    o = dict(tag=tag, section=r["section"], method=r["method"], doc_kind=r["doc_kind"], test_clean="clean", frozen_setting=r["frozen_setting"])
    sc = lambda s, k: ((s["runs"].get(k) or {}).get("score") if s else None)
    dv = [sc(sd, k) for k in DEV]; ts = [sc(stt, k) for k in TEST]
    for k, v in zip(("online", "library", "literature"), dv):
        o[f"dev_{k}"] = f4(v)
    for k, v in zip(TEST, ts):
        o[f"test_{k}"] = f4(v)
    dm = sum(dv) / 3 if all(v is not None for v in dv) else None
    tm = sum(ts) / 5 if all(v is not None for v in ts) else None
    o["dev_mean"], o["test_mean"] = f4(dm), f4(tm)
    o["test_minus_dev"] = f4(tm - dm) if dm is not None and tm is not None else ""
    t = timing.get(tag, {})
    secs = lambda runs: [t[k]["secs"] for k in runs if k in t]
    o["dev_time"] = f"{st.mean(secs(DEV)):.1f} s per run" if secs(DEV) else ""
    o["test_time"] = f"{st.mean(secs(TEST)):.1f} s per run" if secs(TEST) else ""
    nq = sum((stt["runs"].get(k) or {}).get("n", 0) for k in TEST) if stt else 0
    nm = sum(len((stt["runs"].get(k) or {}).get("missing") or []) for k in TEST) if stt else 0
    done = bool(stt and stt["complete"] and sd and sd["complete"])
    ncomplete += done
    o["completeness"] = ("complete" if done else "INCOMPLETE") + f" {nq} q [{nq} q, {nm} missing qids]"
    o["clean_reason"], o["dev_note"], o["notes"] = r["clean_reason"], r["dev_note"], r["notes"]
    out.append(o)
D.mkdir(parents=True, exist_ok=True)
tmp = D / "results_full3b.csv.tmp"
with open(tmp, "w", newline="") as fh:
    w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(out)
tmp.replace(D / "results_full3b.csv")
msg = f"csv {ncomplete}/{len(B.TAGS)} tags complete on all 8 runs"
print(msg)
sys.exit(0 if ncomplete == len(B.TAGS) else 1)
