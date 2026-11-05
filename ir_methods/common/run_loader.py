"""Run loader and BM25 shared by every stage. Reads the public files of a run (events, blocks, questions) from $IRBENCH_RUNS
(default: the repo runs/ folder). Answer keys are opened only when IRBENCH_KEYS=1, which the diagnostic and grid scripts set.
No method reads the key of a run it is scored on; dev fits read dev keys only. Puts the repo root on sys.path so that `import score` finds the repo scorer."""
import json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
PUB = Path(os.environ.get("IRBENCH_RUNS") or REPO / "runs")
sys.path.insert(0, str(REPO))
import score as S
DEV = ["online_shop", "library_maintainers", "literature_review"]
TEST = ["on_call", "due_diligence", "executive_assistants", "help_desk", "textbook_team"]
RUNS = os.environ.get("RUNS", ",".join(DEV + TEST)).split(",")


def load(name):
    d = PUB / name
    run = S.load_run(d)
    key = json.load(open(d / "answer_key.json", encoding="utf-8")) if os.environ.get("IRBENCH_KEYS") == "1" else {}
    qs = [json.loads(l) for l in open(d / "questions.jsonl")]
    blocks = [json.loads(l) for l in open(d / "blocks.jsonl")]
    events = [json.loads(l) for l in open(d / "events.jsonl")]
    return d, run, key, qs, blocks, events


def words(t):
    return re.findall(r"[a-z0-9_]+", t.lower())

class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.docs = [Counter(words(t)) for t in texts]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        self.df = Counter(w for d in self.docs for w in d)
        self.post = defaultdict(list)
        for i, d in enumerate(self.docs):
            for w in d:
                self.post[w].append(i)
        self.n, self.k1, self.b = len(texts), k1, b
    def scores(self, q):
        s = defaultdict(float)
        for w in set(words(q)):
            df = self.df.get(w, 0)
            if not df: continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i in self.post[w]:
                tf = self.docs[i][w]
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s
