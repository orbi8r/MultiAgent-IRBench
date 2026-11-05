"""full3b: shared helpers of the b3 baseline scripts (no training, no tuning, every setting declared in this file).
Env: IRBENCH_WORK IRBENCH_OUT (set by the runner), DEVR TESTR (smokes narrow the run lists), NQ (questions per run, smokes only).
Outputs live under $IRBENCH_OUT/b3subs, b3frozen, b3rows, b3timing, b3done, b3scores so the full3 score_submissions.py and aggregate_full3.py never see them."""
import json, math, os, re, time
from collections import Counter, defaultdict
from pathlib import Path

BASE, OUT = Path(os.environ["IRBENCH_WORK"]), Path(os.environ["IRBENCH_OUT"])
DEVR = os.environ.get("DEVR", "online_shop library_maintainers literature_review").split()
TESTR = os.environ.get("TESTR", "on_call due_diligence executive_assistants help_desk textbook_team").split()
NQ = int(os.environ.get("NQ", 10 ** 9))
SECTION = "B3 Baselines with no training on all 8 runs"
RAW_N, LB_N = 300, 200  # raw lists are cut at 300 like example_bm25 and bm25_ctx_alone; the list builder gets the top 200 like full2 (K 200)
RRF_K, RRF_DEPTH = 60, 1000
STEM_NOTE = "English stop words = sklearn ENGLISH_STOP_WORDS, Porter = nltk PorterStemmer"

# tag: (method, frozen_setting). Order is the row order of the csv.
TAGS = {
    "b3_random_s0": ("uniform random permutation of all pieces per question, seed 0, no list builder", "random.Random(f'0:{run}:{qid}').shuffle, top 300 as is"),
    "b3_random_s1": ("uniform random permutation of all pieces per question, seed 1, no list builder", "random.Random(f'1:{run}:{qid}').shuffle, top 300 as is"),
    "b3_random_s2": ("uniform random permutation of all pieces per question, seed 2, no list builder", "random.Random(f'2:{run}:{qid}').shuffle, top 300 as is"),
    "b3_recent": ("newest piece first, same list for every question, no list builder", "sort by (event order, block index) descending, top 300 as is"),
    "b3_oldest": ("chronological, oldest piece first, same list for every question, no list builder", "sort by (event order, block index) ascending, top 300 as is"),
    "b3_long": ("longest piece first (tokens), same list for every question, no list builder", "sort by tokens descending then chronological, top 300 as is"),
    "b3_short": ("shortest piece first (tokens), same list for every question, no list builder", "sort by tokens ascending then chronological, top 300 as is"),
    "b3_bm25_ws": ("BM25 over plain piece text, lowercase whitespace split, no list builder", "k1 1.2 b 0.75, question text only, top 300 as is"),
    "b3_bm25_word": ("BM25 over plain piece text, lowercase \\w+ tokens, no list builder", "k1 1.2 b 0.75, question text only, top 300 as is"),
    "b3_bm25_stem": ("BM25 over plain piece text, \\w+ tokens, English stop words removed, Porter stem, no list builder", f"k1 1.2 b 0.75, question text only, top 300 as is; {STEM_NOTE}"),
    "b3_bm25_ident": ("BM25 over plain piece text, identifier split (camelCase, snake_case, paths on / . _ -), lowercase, no list builder", "k1 1.2 b 0.75, question text only, top 300 as is"),
    "b3_dense_minilm": ("off the shelf dense, all-MiniLM-L6-v2 cosine over the ctx text, no list builder", "sentence-transformers/all-MiniLM-L6-v2, 512 token cap, fp16, query = question + asker, top 300 as is"),
    "b3_dense_bgeb": ("off the shelf dense, bge-base-en-v1.5 cosine over the ctx text, query instruction, no list builder", "BAAI/bge-base-en-v1.5, 512 token cap, fp16, query prefix 'Represent this sentence for searching relevant passages: ', top 300 as is"),
    "b3_dense_e5b": ("off the shelf dense, e5-base-v2 cosine over the ctx text, query and passage prefixes, no list builder", "intfloat/e5-base-v2, 512 token cap, fp16, 'query: ' and 'passage: ' prefixes, top 300 as is"),
    "b3_dense_qwen3e": ("off the shelf dense, Qwen3-Embedding-0.6B cosine over the ctx text, its query instruction, no list builder", "Qwen/Qwen3-Embedding-0.6B, 512 token cap, fp16, sentence-transformers prompt_name query, top 300 as is"),
    "b3_rrf_stem_bgeb": ("reciprocal rank fusion of b3_bm25_stem and b3_dense_bgeb, no list builder", f"RRF k 60 over the top {RRF_DEPTH} of each list, top 300 as is"),
    "b3_recent_lb": ("newest piece first through the list builder", "b3_recent top 200 into study_helpers.build (build_list.build, LOC_SCAN 30, EARLIEST on, UTIL_A 0)"),
    "b3_bm25_word_lb": ("BM25 \\w+ through the list builder", "b3_bm25_word top 200 into study_helpers.build (30, on)"),
    "b3_bm25_stem_lb": ("BM25 stop words and Porter stem through the list builder", "b3_bm25_stem top 200 into study_helpers.build (30, on)"),
    "b3_dense_bgeb_lb": ("bge-base-en-v1.5 dense through the list builder", "b3_dense_bgeb top 200 into study_helpers.build (30, on)"),
    "b3_dense_qwen3e_lb": ("Qwen3-Embedding-0.6B dense through the list builder", "b3_dense_qwen3e top 200 into study_helpers.build (30, on)"),
    "b3_rrf_stem_bgeb_lb": ("RRF of BM25 stem and bge-base dense through the list builder", "b3_rrf_stem_bgeb top 200 into study_helpers.build (30, on)"),
}
LEX_TAGS = [t for t in TAGS if t.startswith(("b3_random", "b3_recent", "b3_oldest", "b3_long", "b3_short", "b3_bm25"))]
DENSE_TAGS = [t for t in TAGS if t not in LEX_TAGS]
CLEAN_REASON = "B3 baseline, no training and no tuning, setting declared in the task code before any test sub (full3b)"


def plan():
    return [("dev", r) for r in DEVR] + [("test", r) for r in TESTR]


def atomic(path, obj, **kw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    json.dump(obj, open(tmp, "w"), **kw)
    os.replace(tmp, path)


def write_rows_and_freeze(tags):
    """Rows and frozen files first, so a test sub never exists without them. Declared settings, nothing chosen."""
    for t in tags:
        method, setting = TAGS[t]
        atomic(OUT / "b3rows" / f"{t}.json", dict(tag=t, section=SECTION, method=method, doc_kind="FIXED", frozen_setting=setting,
                                                 clean_reason=CLEAN_REASON, dev_note="", notes="one frozen scoring per run, no variant picked by any score"), indent=1)
        f = OUT / "b3frozen" / f"{t}.json"
        if not f.exists():
            atomic(f, dict(tag=t, settings=setting, evidence="declared, no choice", written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1)


def put(tag, split, run, sub, qs):
    assert (OUT / "b3frozen" / f"{tag}.json").exists(), f"{tag}: frozen file must exist before a sub"
    assert all(q["qid"] in sub for q in qs), "sub misses questions"
    assert all(isinstance(v, list) and len(v) <= 400 and all(isinstance(x, str) for x in v) for v in sub.values())
    atomic(OUT / "b3subs" / split / tag / f"{run}.json", sub)


def timing(tag, run, secs, peak_gb=0.0):
    f = OUT / "b3timing" / f"{tag}.json"
    d = json.load(open(f)) if f.exists() else {}
    d[run] = dict(secs=round(float(secs), 2), peak_gb=round(float(peak_gb), 2))
    atomic(f, d, indent=1)


def marker(task, split, run):
    return OUT / "b3done" / f"{task}_{split}_{run}"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# tokenizers: plain piece text and question text go through the same one
_STEM = {}
_ps = None


def _porter(w):
    global _ps
    if _ps is None:
        try:
            from nltk.stem import PorterStemmer
            _ps = PorterStemmer().stem
        except ImportError:
            import snowballstemmer
            _ps = snowballstemmer.stemmer("porter").stemWord
    s = _STEM.get(w)
    if s is None:
        s = _STEM[w] = _ps(w)
    return s


def _stop():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return ENGLISH_STOP_WORDS


_STOP = None
_W = re.compile(r"\w+")
_C1, _C2 = re.compile(r"([a-z0-9])([A-Z])"), re.compile(r"([A-Z]+)([A-Z][a-z])")
_ALN = re.compile(r"[^\W_]+")


def tok_ws(t):
    return t.lower().split()


def tok_word(t):
    return _W.findall(t.lower())


def tok_stem(t):
    global _STOP
    if _STOP is None:
        _STOP = _stop()
    return [_porter(w) for w in _W.findall(t.lower()) if w not in _STOP]


def tok_ident(t):
    t = _C2.sub(r"\1 \2", _C1.sub(r"\1 \2", t))
    return _ALN.findall(t.lower())


TOK = {"ws": tok_ws, "word": tok_word, "stem": tok_stem, "ident": tok_ident}


class BM25:
    """Same formula as bm25_tools.BM and the repo example: idf log(1 + (N - df + 0.5) / (df + 0.5)), k1 1.2, b 0.75, query terms deduplicated."""
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.len = [sum(d.values()) or 1 for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        self.df = Counter(w for d in self.docs for w in d)
        self.post = defaultdict(list)
        for i, d in enumerate(self.docs):
            for w in d:
                self.post[w].append(i)
        self.n, self.k1, self.b = len(docs), k1, b

    def scores(self, q):
        s = defaultdict(float)
        for w in set(q):
            df = self.df.get(w)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i in self.post[w]:
                tf = self.docs[i][w]
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def order_of(s, k):
    """Indices by descending score, ties by piece index (deterministic), only pieces with a score."""
    return sorted(s, key=lambda i: (-s[i], i))[:k]


def bm25_stem_lists(P, qs, depth):
    """Per qid: (ordered piece indices, scores) of BM25 stem over plain text, top `depth`."""
    bm = BM25([tok_stem(p["text"]) for p in P])
    out = {}
    for q in qs:
        s = bm.scores(tok_stem(q["question"]))
        o = order_of(s, depth)
        out[q["qid"]] = (o, [s[i] for i in o])
    return out


def rrf(lists, k=RRF_K):
    s = defaultdict(float)
    first = {}
    for l in lists:
        for r, x in enumerate(l):
            s[x] += 1 / (k + r)  # r from 0, as in the GPU rrf
            first.setdefault(x, len(first))
    o = sorted(s, key=lambda x: (-s[x], first[x]))
    return o, [s[x] for x in o]


def qtext(q):
    return f"{q['question']} (asked by {q['asker']})"
