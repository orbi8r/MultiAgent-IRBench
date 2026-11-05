"""Aggregates the full3 study: joins scores (score_submissions.py), rows and timing of the f3_ and t3_ tags with the imported full2 control scores
($IRBENCH_WORK/ctrl_scores, chosen per tag by control_tags.py) into results_full3.csv, results_full3.md, results_full3_summary.md.
Paired gain = mean over runs of the per question score difference to the primary control, bootstrap 95 percent CI resampling
questions inside each run (2000 draws, seed 0). A branch whose tags are missing (gate false, not run) is listed as skipped."""
import csv, glob, json, os, re, statistics as st
from pathlib import Path
import numpy as np
import control_tags

BASE, OUT = Path(os.environ["IRBENCH_WORK"]), Path(os.environ["IRBENCH_OUT"])
DEV = ["online_shop", "library_maintainers", "literature_review"]
TEST = ["on_call", "due_diligence", "executive_assistants", "help_desk", "textbook_team"]
QT = ["finding", "location", "who", "history", "state", "failure_fix", "decision"]
SC = Path(os.environ.get("SCORES", BASE / "out" / "full3_98_score_all" / "scores"))
CT = Path(os.environ.get("CTRL", BASE / "ctrl_scores"))
BRANCH = [("sect06", r"^f3_sect06_"), ("sect4b", r"^f3_sect4b_"), ("probe", r"^f3_probe_"), ("sp06", r"^f3_sp06_"), ("sp4b", r"^f3_sp4b_"), ("toc", r"^f3_toc")]
FLAGS = [(re.compile(f["pattern"]), f["flag"], f["reason"]) for f in csv.DictReader(open(Path(__file__).resolve().parent / "clean_flags.csv"))]
try:
    from tags import META
except Exception:
    META = {}


def flag(tag):
    for p, fl, why in FLAGS:
        if p.fullmatch(tag):
            return fl, why
    return "unknown", "clean_flags.csv has no pattern for this tag"


def secsort(s):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def load(d, split):
    return {Path(p).stem: json.load(open(p)) for p in glob.glob(str(d / split / "*.json"))}


def pqs(s, runs):
    """{run: {qid: score}} or None when a run is missing."""
    if not s or not all(s["runs"].get(r) and "per_question" in s["runs"][r] for r in runs):
        return None
    return {r: {q: v[1] for q, v in s["runs"][r]["per_question"].items()} for r in runs}


def boot(vecs, B=2000):
    """vecs: per run arrays of per question values. Mean over runs of the run mean with questions resampled inside each run."""
    rng = np.random.default_rng(0)
    cols = [np.asarray(v)[rng.integers(0, len(v), (B, len(v)))].mean(1) for v in vecs]
    m = np.mean(cols, 0)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci(s, runs):
    p = pqs(s, runs)
    return boot([list(p[r].values()) for r in runs]) if p else (None, None)


def paired(sa, sb, runs):
    """(gain, lo, hi) of a minus b on the questions both scored, or None."""
    a, b = pqs(sa, runs), pqs(sb, runs)
    if not a or not b:
        return None
    d = [np.array([a[r][q] - b[r][q] for q in a[r] if q in b[r]]) for r in runs]
    if any(len(x) == 0 for x in d):
        return None
    lo, hi = boot(d)
    return float(np.mean([x.mean() for x in d])), lo, hi


def verdict(g):
    if g is None:
        return "not scored"
    return "gain, CI above 0" if g[1] > 0 else "loss, CI below 0" if g[2] < 0 else "no clear difference, CI spans 0"


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
dev, test = load(SC, "dev"), load(SC, "test")
cdev, ctest = load(CT, "dev"), load(CT, "test")
# keep only the question types the scorer reported, so the tables carry no empty columns
seen = {x[0] for sc in (dev, test, cdev, ctest) for s in sc.values() for v in s["runs"].values() if v for x in (v.get("per_question") or {}).values()}
QT = [q for q in QT if q in seen] or QT
mine = sorted({t for t in set(dev) | set(test) | set(rows) | set(timing) if t.startswith(("f3_", "t3_"))}, key=lambda t: (secsort((rows.get(t) or {}).get("section", "99")), t))
used = sorted({c for t in mine for c in control_tags.control(t)})
CS = "control (imported from full2, scores not rerun)"


def make(tag, imported):
    r = rows.get(tag, {})
    sd, st_ = (cdev.get(tag), ctest.get(tag)) if imported else (dev.get(tag), test.get(tag))
    o = dict(tag=tag, section=CS if imported else r.get("section", "99 Scored tags without row metadata"),
             method=(META.get(tag, ("", tag))[1] if imported else r.get("method", tag)), doc_kind="control" if imported else r.get("doc_kind", ""))
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
    nq = lambda s, runs: sum((s["runs"].get(k) or {}).get("n", 0) for k in runs) if s else 0
    nm = lambda s, runs: sum(len((s["runs"].get(k) or {}).get("missing") or []) for k in runs) if s else 0
    if st_ or sd:
        o["completeness"] = ("complete" if st_ and st_["complete"] and sd and sd["complete"] else "INCOMPLETE") + f" [test {nq(st_, TEST)} q, {nm(st_, TEST)} missing; dev {nq(sd, DEV)} q, {nm(sd, DEV)} missing]"
    else:
        o["completeness"] = "timing only, no sub" if tm else "NO SCORE"
    for split, s, runs in (("dev", sd, DEV), ("test", st_, TEST)):
        for k, v in qmeans(s, runs).items():
            o[f"{split}_{k}"] = v
        o[f"{split}_ci_lo"], o[f"{split}_ci_hi"] = ci(s, runs)
    cs = [] if imported else control_tags.control(tag)
    o["control_tag"] = cs[0] if cs else ""
    o["control_tag2"] = cs[1] if len(cs) > 1 else ""
    for suffix, k in (("", 0), ("2", 1)):
        c = cs[k] if len(cs) > k else None
        for split, s, cd, runs in (("dev", sd, cdev, DEV), ("test", st_, ctest, TEST)):
            g = paired(s, cd.get(c), runs) if c else None
            o[f"paired_gain_{split}{suffix}"], o[f"paired_gain_{split}{suffix}_ci_lo"], o[f"paired_gain_{split}{suffix}_ci_hi"] = g if g else (None, None, None)
    o["dev_note"], o["notes"] = r.get("dev_note", ""), r.get("notes", "")
    return o


out = [make(t, False) for t in mine] + [make(t, True) for t in used]
cols = (["tag", "section", "method", "doc_kind", "test_clean", "frozen_setting", "dev_online", "dev_library", "dev_literature", "dev_mean"] +
        [f"test_{k}" for k in TEST] + ["test_mean", "test_minus_dev", "dev_time", "test_time", "completeness", "clean_reason", "dev_note", "notes"] +
        [f"dev_{k}" for k in QT] + [f"test_{k}" for k in QT] + ["dev_ci_lo", "dev_ci_hi", "test_ci_lo", "test_ci_hi", "control_tag"] +
        [f"paired_gain_{s}{x}{y}" for x in ("", "2") for s in ("dev", "test") for y in ("", "_ci_lo", "_ci_hi")] + ["control_tag2"])
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "results_full3.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader()
    for o in out:
        w.writerow({k: (fmt(v, 4) if isinstance(v, float) else v) for k, v in o.items()})

rng = lambda o, p: f" [{fmt(o[p + '_ci_lo'])}, {fmt(o[p + '_ci_hi'])}]" if o.get(p + "_ci_lo") is not None else ""
gain = lambda o, s, x="": (f"{o['paired_gain_' + s + x]:+.3f} [{o['paired_gain_' + s + x + '_ci_lo']:+.3f}, {o['paired_gain_' + s + x + '_ci_hi']:+.3f}]"
                           if o.get("paired_gain_" + s + x) is not None else "")
gates = {Path(p).parent.name: json.load(open(p)) for p in sorted(glob.glob(str(BASE / "out" / "*" / "gate.json")))}
skipped = {Path(p).parent.name: Path(p).read_text().strip() for p in sorted(glob.glob(str(BASE / "out" / "*" / "SKIPPED.txt")))}
md = ["# full3: structure aware rerankers, frozen from the 3 dev runs, scored on the 5 test runs", "",
      "Controls are the flat full2 rows over the same candidates (imported scores, not rerun). Gain is paired per question against the primary control, "
      "mean over runs, bootstrap 95 percent CI. CI of a mean is over questions inside each run. Test numbers come from one frozen config per tag.", ""]
for br, pat in BRANCH:
    ts = [o for o in out if o["section"] != CS and re.search(pat, o["tag"]) and o["tag"].startswith("f3_")]
    if not ts:
        md += [f"## {br}", "", "no tags: branch not run or skipped by a gate" + "".join(f" ({k}: {v})" for k, v in skipped.items() if br[:4] in k or br in k), ""]
        continue
    md += [f"## {br}", "", "| tag | kind | dev mean [CI] | test mean [CI] | gain dev | gain test | " + " | ".join(QT) + " (test) | control |", "|" + "---|" * (7 + len(QT))]
    for o in ts:
        md.append("| " + " | ".join([o["tag"], o["doc_kind"], fmt(o["dev_mean"]) + rng(o, "dev"), fmt(o["test_mean"]) + rng(o, "test"), gain(o, "dev"), gain(o, "test")] +
                                    [fmt(o[f"test_{k}"]) for k in QT] + [o["control_tag"]]) + " |")
    md.append("")
    for o in ts:
        g = None if o["paired_gain_test"] is None else (o["paired_gain_test"], o["paired_gain_test_ci_lo"], o["paired_gain_test_ci_hi"])
        gd = None if o["paired_gain_dev"] is None else (o["paired_gain_dev"], o["paired_gain_dev_ci_lo"], o["paired_gain_dev_ci_hi"])
        md.append(f"Decision {o['tag']}: test {verdict(g)}" + (f" ({gain(o, 'test')})" if g else "") + f"; dev {verdict(gd)}" + (f" ({gain(o, 'dev')})" if gd else "") + ".")
    md.append("")
if gates or skipped:
    md += ["## gates", ""] + [f"- {k}: go {v.get('go')} {v.get('why', '')}"[:400] for k, v in gates.items()] + [f"- {k}: SKIPPED {v}"[:300] for k, v in skipped.items()] + [""]
md += [f"## {CS}", "", "| tag | dev mean [CI] | test mean [CI] | " + " | ".join(QT) + " (test) |", "|" + "---|" * (3 + len(QT))]
for o in [o for o in out if o["section"] == CS]:
    md.append("| " + " | ".join([o["tag"], fmt(o["dev_mean"]) + rng(o, "dev"), fmt(o["test_mean"]) + rng(o, "test")] + [fmt(o[f"test_{k}"]) for k in QT]) + " |")
(OUT / "results_full3.md").write_text("\n".join(md) + "\n")

scored = sorted([o for o in out if o["test_mean"] is not None], key=lambda o: -o["test_mean"])
sm = ["# full3 summary, scored tags by test mean", "", "| tag | clean | dev mean | test mean | CI low | CI high | gain test | control | " + " | ".join(QT) + " |", "|" + "---|" * (8 + len(QT))]
for o in scored:
    sm.append("| " + " | ".join([o["tag"], o["test_clean"], fmt(o["dev_mean"]), fmt(o["test_mean"]), fmt(o["test_ci_lo"]), fmt(o["test_ci_hi"]), gain(o, "test"), o["control_tag"]] +
                                [fmt(o[f"test_{k}"]) for k in QT]) + " |")
(OUT / "results_full3_summary.md").write_text("\n".join(sm) + "\n")
print(len(mine), "full3 tags,", len(used), "control rows,", len(scored), "with a complete test score")
