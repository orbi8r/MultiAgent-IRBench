"""Shared helpers of the full2 study steps: the dev then test plan, the list builder, frozen files, subs, rows,
timing and the per tag score cache. No answer key is opened here; only the scoring step (score_submissions.py) reads keys.
Env: IRBENCH_WORK IRBENCH_OUT (set by the runner), DEVR TESTR (run lists, smokes narrow them), NQ (questions per run, smokes only)."""
import json, os, time
from collections import defaultdict
from pathlib import Path

BASE, OUT = Path(os.environ["IRBENCH_WORK"]), Path(os.environ["IRBENCH_OUT"])
DEVR = os.environ.get("DEVR", "online_shop library_maintainers literature_review").split()
TESTR = os.environ.get("TESTR", "on_call due_diligence executive_assistants help_desk textbook_team").split()
NQ = int(os.environ.get("NQ", 10 ** 9))
TASK = os.environ.get("TASKNAME", OUT.name)
INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")
PUBFILES = [f"pub_{i:02d}.jsonl" for i in range(15)]  # batch 1 of the public rows, 4964 rows, what ftpub was trained on
os.environ.setdefault("EARLIEST", "1"); os.environ.setdefault("LOC_SCAN", "30"); os.environ.setdefault("UTIL_A", "0")


def script(name):
    """Path of a script in another stage folder (the stage folders sit next to common/), for subprocess calls."""
    return str(next(Path(__file__).resolve().parents[1].glob(f"*/{name}")))


def plan():
    """(split, run) pairs: the dev runs first, then the test runs."""
    return [("dev", r) for r in DEVR] + [("test", r) for r in TESTR]


def questions(run):
    return [json.loads(l) for l in open(f"data/{run}/questions.jsonl")][:NQ]


def builder(name):
    """Keyless list builder state of a run: questions, Files index, run dict, pieces."""
    from run_loader import load
    from file_provenance import Files
    import build_list
    build_list.SCAN, build_list.EARLIEST = 30, True
    d, run, key, qs, blocks, events = load(name)
    P = [json.loads(l) for l in open(f"data/{name}/pieces.jsonl")]
    F = Files(events, P); F.by_event = defaultdict(list); F.tokens = run["tokens"]
    for p in P:
        F.by_event[p["event"]].append(p["id"])
    return questions(name), F, run, P


def from_run(r):
    """List builder file index of an arch1.Run (as write_pool_submissions.py does it): returns F ready for build()."""
    import build_list
    build_list.SCAN, build_list.EARLIEST = 30, True
    F = r.F; F.by_event = defaultdict(list); F.tokens = r.run["tokens"]
    for p in r.P:
        F.by_event[p["event"]].append(p["id"])
    return F


def build(F, q, run, cands, scores):
    import build_list
    return build_list.build(F, q, list(cands), list(scores), run["text"])


def ranked(cands, scores):
    return [x for _, x in sorted(zip(scores, cands), key=lambda t: -t[0])]


def atomic(path, obj, **kw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    json.dump(obj, open(tmp, "w"), **kw)
    os.replace(tmp, path)


def freeze(tag, settings, evidence="declared, no choice"):
    """Write $IRBENCH_OUT/frozen/<tag>.json once, before any test sub of the tag. A retry reads the file back instead of rewriting it."""
    f = OUT / "frozen" / f"{tag}.json"
    if f.exists():
        return json.load(open(f))["settings"]
    atomic(f, dict(tag=tag, task=TASK, settings=settings, evidence=evidence, written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1)
    return settings


def put(tag, split, run, sub, qs):
    assert split in ("dev", "test")
    if split == "test":
        assert (OUT / "frozen" / f"{tag}.json").exists(), f"{tag}: frozen file must exist before a test sub"
    assert all(q["qid"] in sub for q in qs), "sub misses questions"
    assert all(isinstance(v, list) and len(v) <= 400 for v in sub.values())
    atomic(OUT / "subs" / split / tag / f"{run}.json", sub)


def have(tag, split, run):
    return (OUT / "subs" / split / tag / f"{run}.json").exists()


def row(tag, section, method, doc_kind, frozen_setting, dev_note="", notes=""):
    atomic(OUT / "rows" / f"{tag}.json", dict(tag=tag, section=section, method=method, doc_kind=doc_kind,
                                              frozen_setting=frozen_setting, clean_reason="", dev_note=dev_note, notes=notes), indent=1)


def timing(tag, run, secs, peak_gb=0.0, **parts):
    f = OUT / "timing" / f"{tag}.json"
    d = json.load(open(f)) if f.exists() else {}
    d[run] = dict(secs=round(float(secs), 2), peak_gb=round(float(peak_gb), 2), **parts)
    atomic(f, d, indent=1)


def sub_from_gpu(name, gtag, model, mode="build"):
    """Sub of a gpu file data/<run>/gpu_<gtag>.json: build = list builder over the reader order, raw = reader order, first = cands order."""
    G = json.load(open(f"data/{name}/gpu_{gtag}.json"))
    qs, F, run, P = builder(name)
    cands = G["cands"]
    sub = {}
    for q in qs:
        c = cands[q["qid"]]
        if mode == "first":
            sub[q["qid"]] = c[:200]
        else:
            s = G["scores"][model][q["qid"]]
            sub[q["qid"]] = ranked(c, s) if mode == "raw" else build(F, q, run, c, s)
    return qs, sub, G


def peak_gb():
    import torch
    return torch.cuda.max_memory_allocated() / 2 ** 30


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
