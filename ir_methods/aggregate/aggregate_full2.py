"""Aggregates the full2 study: joins scores (score_submissions.py), rows and timing of every tag into results_full2.csv, results_full2.md,
results_full2_runqtype.csv (per run x per qtype, long form), results_full2_runqtype.md and results_full2_summary.md.
Header is the results_test5.csv header plus qtype means, bootstrap 95 percent CI over questions and timing columns.
test5 rows are not read."""
import csv, glob, json, os, re, statistics as st
from pathlib import Path
import numpy as np
BASE, OUT = Path(os.environ["IRBENCH_WORK"]), Path(os.environ["IRBENCH_OUT"])
DEV = ["online_shop", "library_maintainers", "literature_review"]
TEST = ["on_call", "due_diligence", "executive_assistants", "help_desk", "textbook_team"]
QT = ["who", "location", "finding", "history"]
SC = Path(os.environ.get("SCORES", BASE / "out" / "full2_98_score_all" / "scores"))
FLAGS = [(re.compile(f["pattern"]), f["flag"], f["reason"]) for f in csv.DictReader(open(Path(__file__).resolve().parent / "clean_flags.csv"))]


def flag(tag):
    for p, fl, why in FLAGS:
        if p.fullmatch(tag):
            return fl, why
    return "unknown", "clean_flags.csv has no pattern for this tag"


def secsort(s):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def load_scores(split):
    return {Path(p).stem: json.load(open(p)) for p in glob.glob(str(SC / split / "*.json"))}


def boot(s, runs, B=2000):
    """Mean over runs of the run mean, resampling questions inside every run; rng seed fixed so the CI is reproducible."""
    if not s or not all(s["runs"].get(r) and "per_question" in s["runs"][r] for r in runs):
        return None, None
    rng = np.random.default_rng(0)
    cols = []
    for r in runs:
        v = np.array([x[1] for x in s["runs"][r]["per_question"].values()])
        cols.append(v[rng.integers(0, len(v), (B, len(v)))].mean(1))
    m = np.mean(cols, 0)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def qmeans(s, runs):
    out = {}
    for qt in QT:
        v = [x[1] for r in runs if s and s["runs"].get(r) and "per_question" in s["runs"][r] for x in s["runs"][r]["per_question"].values() if x[0] == qt]
        out[qt] = sum(v) / len(v) if v else None
    return out


rows = {}
for f in glob.glob(str(BASE / "out" / "*" / "rows" / "*.json")):
    r = json.load(open(f)); rows[r["tag"]] = r
timing = {}
for f in glob.glob(str(BASE / "out" / "*" / "timing" / "*.json")):
    timing[Path(f).stem] = json.load(open(f))
dev, test = load_scores("dev"), load_scores("test")
tags = sorted(set(dev) | set(test) | set(rows) | set(timing), key=lambda t: (secsort((rows.get(t) or {}).get("section", "99")), t))
out, long_ = [], []
for tag in tags:
    r = rows.get(tag, {})
    sd, st_ = dev.get(tag), test.get(tag)
    o = dict(tag=tag, section=r.get("section", "99 Scored tags without row metadata"), method=r.get("method", tag), doc_kind=r.get("doc_kind", ""))
    o["test_clean"], o["clean_reason"] = flag(tag)
    o["frozen_setting"] = r.get("frozen_setting", "")
    dv = [(sd["runs"][k] or {}).get("score") if sd and sd["runs"].get(k) else None for k in DEV]
    ts = [(st_["runs"][k] or {}).get("score") if st_ and st_["runs"].get(k) else None for k in TEST]
    for k, v in zip(("online", "library", "literature"), dv):
        o[f"dev_{k}"] = v
    o["dev_mean"] = sum(dv) / 3 if all(x is not None for x in dv) else None
    for k, v in zip(TEST, ts):
        o[f"test_{k}"] = v
    o["test_mean"] = sum(ts) / 5 if all(x is not None for x in ts) else None
    o["test_minus_dev"] = o["test_mean"] - o["dev_mean"] if o["test_mean"] is not None and o["dev_mean"] is not None else None
    tm = timing.get(tag, {})
    tsec = lambda runs: [tm[k]["secs"] for k in runs if k in tm and tm[k].get("secs") is not None]
    o["dev_time"] = f"{st.mean(tsec(DEV)):.1f} s per run" if tsec(DEV) else ""
    o["test_time"] = f"{st.mean(tsec(TEST)):.1f} s per run" if tsec(TEST) else ""
    o["time_secs_mean"] = st.mean(tsec(DEV + TEST)) if tsec(DEV + TEST) else None
    pk = [tm[k].get("peak_gb") for k in tm if isinstance(tm[k], dict) and tm[k].get("peak_gb")]
    o["time_peak_gb"] = max(pk) if pk else None
    nq = lambda s, runs: sum((s["runs"].get(k) or {}).get("n", 0) for k in runs) if s else 0
    nm = lambda s, runs: sum(len((s["runs"].get(k) or {}).get("missing") or []) for k in runs) if s else 0
    if st_ or sd:
        o["completeness"] = ("complete" if st_ and st_["complete"] and sd and sd["complete"] else "INCOMPLETE") + f" [test {nq(st_, TEST)} q, {nm(st_, TEST)} missing; dev {nq(sd, DEV)} q, {nm(sd, DEV)} missing]"
    else:
        o["completeness"] = "timing only, no sub" if tm else "NO SCORE"
    for split, s, runs in (("dev", sd, DEV), ("test", st_, TEST)):
        q = qmeans(s, runs)
        for k in QT:
            o[f"{split}_{k}"] = q[k]
        o[f"{split}_ci_lo"], o[f"{split}_ci_hi"] = boot(s, runs)
        if s:
            for run in runs:
                x = (s["runs"].get(run) or {})
                for qt, v in (x.get("by_qtype") or {}).items():
                    long_.append(dict(tag=tag, split=split, run=run, qtype=qt, score=v["score"], n=v["n"]))
                if "score" in x:
                    long_.append(dict(tag=tag, split=split, run=run, qtype="all", score=x["score"], n=x["n"]))
    o["dev_note"], o["notes"] = r.get("dev_note", ""), r.get("notes", "")
    out.append(o)
cols = (["tag", "section", "method", "doc_kind", "test_clean", "frozen_setting", "dev_online", "dev_library", "dev_literature", "dev_mean"] +
        [f"test_{k}" for k in TEST] + ["test_mean", "test_minus_dev", "dev_time", "test_time", "completeness", "clean_reason", "dev_note", "notes"] +
        [f"dev_{k}" for k in QT] + [f"test_{k}" for k in QT] + ["dev_ci_lo", "dev_ci_hi", "test_ci_lo", "test_ci_hi", "time_secs_mean", "time_peak_gb"])
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "results_full2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader()
    for o in out:
        w.writerow({k: (fmt(v, 4) if isinstance(v, float) else v) for k, v in o.items()})
with open(OUT / "results_full2_runqtype.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["tag", "split", "run", "qtype", "score", "n"]); w.writeheader()
    for x in long_:
        w.writerow({**x, "score": f"{x['score']:.4f}"})

md = ["# full2: final study, frozen methods scored on dev and test", "",
      "Every method was frozen from the 3 dev runs (online_shop, library_maintainers, literature_review) or from public data before its test runs were read. "
      "Test numbers are on the 5 test runs, v5_abs with the repo scorer. CI is a bootstrap 95 percent interval over questions inside each run (2000 draws), "
      "taken on the mean over runs. Dev subs of fitted methods are leave one run out. Timing is seconds per run on one RTX 2080 Ti, model loads excluded.", ""]
for sec in sorted({o["section"] for o in out}, key=secsort):
    md += [f"## {sec}", "", "| method (tag) | kind | clean | dev mean [CI] | " + " | ".join(t.replace("_", " ") for t in TEST) + " | test mean [CI] | who | location | finding | history | time s | complete | notes |", "|" + "---|" * 16]
    for o in [o for o in out if o["section"] == sec]:
        ci = lambda p: f" [{fmt(o[p + '_ci_lo'])}, {fmt(o[p + '_ci_hi'])}]" if o[p + "_ci_lo"] is not None else ""
        md.append("| " + " | ".join([f"{o['method']} ({o['tag']})", o["doc_kind"], o["test_clean"], fmt(o["dev_mean"]) + ci("dev")] +
                  [fmt(o[f"test_{k}"]) for k in TEST] + [fmt(o["test_mean"]) + ci("test")] + [fmt(o[f"test_{k}"]) for k in QT] +
                  [fmt(o["time_secs_mean"], 1), o["completeness"], (o["dev_note"] + " " + o["notes"]).strip().replace("|", "/")]) + " |")
    md.append("")
(OUT / "results_full2.md").write_text("\n".join(md))

scored = sorted([o for o in out if o["test_mean"] is not None], key=lambda o: -o["test_mean"])
sm = ["# full2 summary, scored tags by test mean", "", "| tag | clean | dev mean | test mean | CI low | CI high | who | location | finding | history | time s |", "|" + "---|" * 11]
for o in scored:
    sm.append("| " + " | ".join([o["tag"], o["test_clean"], fmt(o["dev_mean"]), fmt(o["test_mean"]), fmt(o["test_ci_lo"]), fmt(o["test_ci_hi"])] + [fmt(o[f"test_{k}"]) for k in QT] + [fmt(o["time_secs_mean"], 1)]) + " |")
(OUT / "results_full2_summary.md").write_text("\n".join(sm))
RUNS = DEV + TEST
pv = ["# full2 per run x per qtype score, top 15 tags by test mean (all tags in results_full2_runqtype.csv)", "", "| tag | " + " | ".join(f"{r} {q}" for r in TEST for q in QT) + " |", "|" + "---|" * (1 + 4 * len(TEST))]
idx = {(x["tag"], x["split"], x["run"], x["qtype"]): x["score"] for x in long_}
for o in scored[:15]:
    pv.append("| " + " | ".join([o["tag"]] + [fmt(idx.get((o["tag"], "test", r, q))) for r in TEST for q in QT]) + " |")
(OUT / "results_full2_runqtype.md").write_text("\n".join(pv))
print(len(out), "tags,", len(scored), "with a complete test score")
