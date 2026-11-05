"""The whole fast method as one process, timed end to end per run (no keys, nothing fitted on the runs).
  load record -> tokenize (stem cache) -> novelty -> sparse BM25 indexes (scipy) -> first stage for all questions at once
  (fuse0: RRF of BM25 ctx, plain, raw identifiers, novelty pruned, feedback, clause split, question type) -> reader
  (public trace Qwen3 Reranker 0.6B LoRA, merged, fp16, sdpa, length sorted batches, one model copy per GPU, threads)
  -> list builder (dedup, earliest copy, location provenance solver).
Reader configs run one after another on the same loaded models; each is reported as total = shared stages + its own.
    python timed_run.py RUN ADAPTER CONFIGS OUT     CONFIGS e.g. K50,K20,X100,K50t,K100t (t = span trimmed input)
Writes OUT = {times: {...}, lists: {config: {qid: [ids]}}}."""
import json, math, os, re, sys, time, threading
from collections import Counter, defaultdict
T0 = time.time()
if os.environ.get("NO_TRITON"):
    sys.modules["triton"] = None
import numpy as np
import scipy.sparse as sp

run, adapter, configs, outp = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
D = f"data/{run}"
TM = {}


def lap(k, t):
    TM[k] = round(time.time() - t, 3)


t = time.time()
P = [json.loads(l) for l in open(f"{D}/pieces.jsonl")]
Q = [json.loads(l) for l in open(f"{D}/questions.jsonl")][:int(os.environ.get("NQ", 10**9))]  # NQ caps the question count, for smoke runs only
events = [json.loads(l) for l in open(f"{D}/events.jsonl")]
ids = [p["id"] for p in P]; N = len(P)
lap("load_record", t)

t = time.time()
from contextual_pieces import toks  # stemmed words, identifiers split, stem cache
ID = re.compile(r"#?\w*\d\w*|\w+[_./-]\w[\w./-]*")
WR = re.compile(r"[a-z0-9_]+")


def ident(text):
    out = []
    for m in ID.findall(text):
        m = m.lower().lstrip("#")
        if len(m) >= 3:
            out.append(m)
            parts = re.split(r"([-_./])", m)
            out += ["".join(parts[:k]) for k in range(3, len(parts) - 1, 2)]
    return out


def _tok3(chunk):
    return [(toks(c), toks(t), WR.findall(t.lower()) + ident(t)) for c, t in chunk]


NPROC = int(os.environ.get("NPROC", 1))
items = [(p["ctx"], p["text"]) for p in P]
if NPROC > 1:  # fork workers; each keeps its own stem cache
    import multiprocessing as mp
    with mp.get_context("fork").Pool(NPROC) as pool:
        res = [x for part in pool.map(_tok3, [items[i::NPROC] for i in range(NPROC)]) for x in part]
    order = [i for k in range(NPROC) for i in range(k, len(items), NPROC)]
    tri = [None] * len(items)
    for i, x in zip(order, res):
        tri[i] = x
else:
    tri = _tok3(items)
tc, tp, traw = [x[0] for x in tri], [x[1] for x in tri], [x[2] for x in tri]
lap("tokenize", t)
t = time.time()
from novelty import novelty
nov = novelty(P); keep = np.array([nov[x] > 0.2 for x in ids])
lap("novelty", t)


class SBM:
    """BM25 as a sparse V x N weight matrix; a query is a binary term vector, so all questions score in one product.
    A pruned index reuses the full term count matrix with the dropped rows zeroed (idf and mean length over kept rows)."""
    def __init__(self, docs=None, mask=None, k1=1.2, b=0.75, base=None):
        if base is None:
            self.voc = {}
            rows, cols, vals = [], [], []
            for i, d in enumerate(docs):
                for w, n in Counter(d).items():
                    rows.append(i); cols.append(self.voc.setdefault(w, len(self.voc))); vals.append(n)
            self.tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.voc)), dtype=np.float32)
        else:
            self.voc, self.tf = base.voc, base.tf
        tf = self.tf if mask is None else sp.diags(mask.astype(np.float32)) @ self.tf
        tf.eliminate_zeros()
        lens = np.asarray(tf.sum(1)).ravel()
        n = int((lens > 0).sum())
        df = np.bincount(tf.indices, minlength=len(self.voc))
        self.n, self.df = n, df
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = lens[lens > 0].mean()
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * lens[tf.row] / avg)
        w = self.idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm)
        self.W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # V x N

    def qmat(self, qs):
        r, c = [], []
        for j, q in enumerate(qs):
            for w in set(q):
                if w in self.voc:
                    r.append(j); c.append(self.voc[w])
        return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(qs), len(self.voc)))

    def scores(self, qs):
        return (self.qmat(qs) @ self.W).toarray()  # nq x N


def top(S, k=1000, mask=None):
    S = S.copy()
    if mask is not None:
        S[:, ~mask] = -1
    S[S <= 0] = -1
    idx = np.argpartition(-S, min(k, S.shape[1] - 1), axis=1)[:, :k]
    out = []
    for j in range(S.shape[0]):
        row = idx[j][S[j, idx[j]] > 0]
        out.append(list(row[np.argsort(-S[j, row])]))
    return out


def rrf(*lists, k=60):
    s = defaultdict(float)
    for l in lists:
        for r, i in enumerate(l):
            s[i] += 1 / (k + r)
    return sorted(s, key=lambda i: -s[i])


t = time.time()
bc, bp, br = SBM(tc), SBM(tp), SBM(traw)
nc, np_ = SBM(mask=keep, base=bc), SBM(mask=keep, base=bp)
lap("index", t)

t = time.time()
QX = {"who": "git commit author committed shift", "location": "mv moved file path saved folder", "history": "first earlier originally"}


def clauses(q):
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+(?=in which|what|who|where|when|how|which)|\bso that\b|\bafter\b|\bbefore\b", q)
    return [p for p in parts if len(p.split()) >= 3]


qt = [toks(q["question"]) for q in Q]
Sc, Sp = bc.scores(qt), bp.scores(qt)
Lc, Lp = top(Sc), top(Sp)
Lr = top(br.scores([WR.findall(q["question"].lower()) + ident(q["question"]) for q in Q]))
Lnc, Lnp = top(nc.scores(qt), mask=keep), top(np_.scores(qt), mask=keep)
rm3q = []
for j in range(len(Q)):
    base = rrf(Lc[j], Lp[j])[:10]
    tf = Counter()
    for i in base:
        for w, c in Counter(tc[i]).items():
            tf[w] += c / len(tc[i]) * math.log(bc.n / bc.df[bc.voc[w]])
    rm3q.append(qt[j] + [w for w, _ in tf.most_common(25)])
Lrm = top(bc.scores(rm3q))
Lqt = top(bc.scores([qt[j] + toks(QX.get(q["qtype"], "")) for j, q in enumerate(Q)]))
cl = [clauses(q["question"]) for q in Q]
flat = [toks(c) for cs in cl for c in cs if len(cs) > 1]
Lcl = top(bc.scores(flat)) if flat else []
fuse, k = {}, 0
for j, q in enumerate(Q):
    if len(cl[j]) > 1:
        dec = rrf(*Lcl[k:k + len(cl[j])]); k += len(cl[j])
    else:
        dec = rrf(Lc[j], Lp[j])
    fuse[q["qid"]] = [ids[i] for i in rrf(Lc[j], Lp[j], Lr[j], Lnc[j], Lrm[j], dec, Lqt[j])[:200]]
lap("first_stage", t)

t = time.time()
import build_list
from file_provenance import Files
build_list.SCAN, build_list.EARLIEST = 30, True
F = Files(events, P); F.by_event = defaultdict(list); F.tokens = {p["id"]: p["tokens"] for p in P}
for p in P:
    F.by_event[p["event"]].append(p["id"])
text = {p["id"]: p["text"] for p in P}
lap("builder_index", t)
if os.environ.get("FIRST_ONLY"):  # CPU check of the first stage alone
    json.dump(dict(times=TM, fuse=fuse), open(outp, "w")); print(json.dumps(TM)); sys.exit(0)

t = time.time()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
G = torch.cuda.device_count()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="sdpa")
base = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
import copy
models = [base.to("cuda:0")] + [copy.deepcopy(base).to(f"cuda:{g}") for g in range(1, G)]
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
pre = tok.encode("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                 "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")
heads = {q["qid"]: tok.encode(f"<Instruct>: {INSTR}\n<Query>: {q['question']} (asked by {q['asker']})\n<Document>: ",
                              add_special_tokens=False) for q in Q}
for g, m in enumerate(models):  # warm up each GPU
    with torch.no_grad():
        m(**tok.pad({"input_ids": [pre + [1] * 64 + suf] * 4}, return_tensors="pt").to(f"cuda:{g}"), logits_to_keep=1)
torch.cuda.synchronize()
lap("load_reader", t)
pid = {p["id"]: p for p in P}
qtok = {q["qid"]: set(toks(q["question"])) for q in Q}
CAP = 480


def doc_ids(qid, x, trim):
    c = pid[x]["ctx"]
    if trim:
        head, _, rest = c.partition("\n")
        w = rest.split()
        if len(w) > 110:  # keep the header (who, step, producing command) + the 100 word window richest in question terms
            hit = [bool(set(toks(v)) & qtok[qid]) for v in w]
            cs = np.cumsum([0] + hit)
            s0 = int(np.argmax([cs[i + 100] - cs[i] for i in range(0, len(w) - 99, 10)])) * 10
            c = head[:400] + "\n" + " ".join(w[s0:s0 + 100])
    return (heads[qid] + tok.encode(c, add_special_tokens=False))[:CAP]


def score_prefix(pairs, trim):
    """Same scores, but the shared prompt prefix (system, instruction, question) of each question is run once and its
    key/value cache reused for all that question's pieces (right padded, last real token's yes/no logits)."""
    from transformers import DynamicCache
    byq = defaultdict(list)
    for i, (a, b) in enumerate(pairs):
        byq[a].append(i)
    out = [0.0] * len(pairs)
    qids = sorted(byq, key=lambda a: -len(byq[a]))
    shards = [qids[g::G] for g in range(G)]

    def work(g):
        m = models[g]; dev = f"cuda:{g}"
        Wyn = m.lm_head.weight[[no, yes]]
        with torch.no_grad():
            for a in shards[g]:
                pre_ids = pre + heads[a]; Lp = len(pre_ids)
                pkv = m(input_ids=torch.tensor([pre_ids], device=dev), use_cache=True).past_key_values
                if hasattr(pkv, "layers"):
                    legacy = [(L.keys, L.values) for L in pkv.layers]
                elif hasattr(pkv, "key_cache"):
                    legacy = list(zip(pkv.key_cache, pkv.value_cache))
                else:
                    legacy = list(pkv)
                docs = {i: doc_ids(a, pairs[i][1], trim)[len(heads[a]):] + suf for i in byq[a]}
                order = sorted(byq[a], key=lambda i: len(docs[i]))
                for s in range(0, len(order), 64):
                    ch = order[s:s + 64]; n = len(ch); Ld = max(len(docs[i]) for i in ch)
                    ids_ = torch.full((n, Ld), tok.pad_token_id, device=dev); am = torch.zeros((n, Lp + Ld), dtype=torch.long, device=dev)
                    am[:, :Lp] = 1
                    last = []
                    for r, i in enumerate(ch):
                        d = docs[i]; ids_[r, :len(d)] = torch.tensor(d, device=dev); am[r, Lp:Lp + len(d)] = 1; last.append(len(d) - 1)
                    cache = DynamicCache()
                    for li, (k, v) in enumerate(legacy):
                        cache.update(k.expand(n, -1, -1, -1).contiguous(), v.expand(n, -1, -1, -1).contiguous(), li)
                    pos = torch.arange(Lp, Lp + Ld, device=dev).unsqueeze(0).expand(n, -1)
                    h = m.model(input_ids=ids_, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
                    hl = h[torch.arange(n, device=dev), torch.tensor(last, device=dev)]
                    lg = (hl @ Wyn.T).float()
                    lp = torch.log_softmax(lg, 1)[:, 1].tolist()
                    for i, v in zip(ch, lp):
                        out[i] = v
    th = [threading.Thread(target=work, args=(g,)) for g in range(G)]
    [x.start() for x in th]; [x.join() for x in th]
    torch.cuda.synchronize()
    return out


def score(pairs, trim):
    bodies = [doc_ids(a, b, trim) for a, b in pairs]
    idx = sorted(range(len(pairs)), key=lambda i: len(bodies[i]))
    out = [0.0] * len(pairs)
    shards = [idx[g::G] for g in range(G)]

    def work(g):
        with torch.no_grad():
            sh = shards[g]
            for s in range(0, len(sh), 64):
                ch = sh[s:s + 64]
                b = tok.pad({"input_ids": [pre + bodies[i] + suf for i in ch]}, padding=True, return_tensors="pt").to(f"cuda:{g}")
                lg = models[g](**b, logits_to_keep=1).logits[:, -1, :]
                lp = torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], 1).float(), 1)[:, 1].tolist()
                for i, v in zip(ch, lp):
                    out[i] = v
    th = [threading.Thread(target=work, args=(g,)) for g in range(G)]
    [x.start() for x in th]; [x.join() for x in th]
    torch.cuda.synchronize()
    return out


STU = os.environ.get("STUDENT")
if STU:
    from transformers import AutoModelForSequenceClassification
    t = time.time()
    stok = AutoTokenizer.from_pretrained(STU)
    smods = [AutoModelForSequenceClassification.from_pretrained(STU).half().to(f"cuda:{g}").eval() for g in range(G)]
    TM["load_student"] = round(time.time() - t, 3)


def stu_score(pairs):
    qtx = {q["qid"]: f"{q['question']} (asked by {q['asker']})" for q in Q}
    idx = sorted(range(len(pairs)), key=lambda i: len(pid[pairs[i][1]]["ctx"]))
    out = [0.0] * len(pairs); shards = [idx[g::G] for g in range(G)]

    def work(g):
        with torch.no_grad():
            sh = shards[g]
            for s in range(0, len(sh), 128):
                ch = sh[s:s + 128]
                e = stok([qtx[pairs[i][0]] for i in ch], [pid[pairs[i][1]]["ctx"] for i in ch], truncation="only_second",
                         max_length=512, padding=True, return_tensors="pt").to(f"cuda:{g}")
                for i, v in zip(ch, smods[g](**e).logits[:, 0].float().tolist()):
                    out[i] = v
    th = [threading.Thread(target=work, args=(g,)) for g in range(G)]
    [x.start() for x in th]; [x.join() for x in th]
    torch.cuda.synchronize()
    return out


LISTS = {}
for cfg in configs:
    t = time.time()
    usep = cfg.endswith("p"); c1 = cfg.rstrip("p")
    trim = c1.endswith("t"); c0 = c1.rstrip("t")
    sc_fn = score_prefix if usep else score
    if c0.startswith("S"):
        parts = [int(x) for x in c0[1:].split("_")]
        k1, n = parts[0], parts[1]; n2 = parts[2] if len(parts) > 2 else None  # S<k1>_<n>[_<n2>]: early exit to n2
        ts = time.time()
        sp_ = [(q["qid"], x) for q in Q for x in fuse[q["qid"]][:k1]]
        sv = stu_score(sp_); SS = defaultdict(dict)
        for (a, b), s_ in zip(sp_, sv):
            SS[a][b] = s_
        TM[f"{cfg}_student"] = round(time.time() - ts, 3)
        pairs = [(q["qid"], x) for q in Q for x in sorted(SS[q["qid"]], key=lambda x: -SS[q["qid"]][x])[:n]]
        v = sc_fn(pairs, trim); SC = defaultdict(dict)
        for (a, b), s_ in zip(pairs, v):
            SC[a][b] = s_
        if n2:
            more = [(q["qid"], x) for q in Q if max(SC[q["qid"]].values()) < -0.05
                    for x in sorted(SS[q["qid"]], key=lambda x: -SS[q["qid"]][x])[n:n2]]
            for (a, b), s_ in zip(more, sc_fn(more, trim) if more else []):
                SC[a][b] = s_
            TM[f"{cfg}_extended_q"] = len({a for a, _ in more})
        for q in Q:  # the rest of the student's list follows below the reader scored pieces, in student order
            lo = min(SC[q["qid"]].values()) - 1
            done_ = set(SC[q["qid"]])
            for r_, x in enumerate([x for x in sorted(SS[q["qid"]], key=lambda x: -SS[q["qid"]][x]) if x not in done_]):
                SC[q["qid"]][x] = lo - r_ * 1e-3
    elif c0.startswith("K"):
        K = int(c0[1:]); pairs = [(q["qid"], x) for q in Q for x in fuse[q["qid"]][:K]]
        v = sc_fn(pairs, trim); SC = defaultdict(dict)
        for (a, b), s in zip(pairs, v):
            SC[a][b] = s
    else:  # X<k>: early exit, score top 20; questions whose best log p(yes) < -0.05 get ranks 20..k scored too
        K = int(c0[1:]); pairs = [(q["qid"], x) for q in Q for x in fuse[q["qid"]][:20]]
        v = sc_fn(pairs, trim); SC = defaultdict(dict)
        for (a, b), s in zip(pairs, v):
            SC[a][b] = s
        more = [(q["qid"], x) for q in Q if max(SC[q["qid"]].values()) < -0.05 for x in fuse[q["qid"]][20:K]]
        for (a, b), s in zip(more, sc_fn(more, trim) if more else []):
            SC[a][b] = s
        TM[f"{cfg}_extended_q"] = len({a for a, _ in more})
    TM[f"{cfg}_reader"] = round(time.time() - t, 3)  # includes the student when there is one
    TM[f"{cfg}_pairs"] = sum(len(v) for v in SC.values())
    t2 = time.time()
    LISTS[cfg] = {}
    for q in Q:
        c = list(SC[q["qid"]]); LISTS[cfg][q["qid"]] = build_list.build(F, q, c, [SC[q["qid"]][x] for x in c], text)
    TM[f"{cfg}_build"] = round(time.time() - t2, 3)
shared = sum(TM[k] for k in ("load_record", "tokenize", "novelty", "index", "first_stage", "builder_index"))
TM["shared_cpu_s"] = round(shared, 2)
for cfg in configs:
    TM[f"{cfg}_total_excl_model_load"] = round(shared + TM[f"{cfg}_reader"] + TM[f"{cfg}_build"], 2)
TM["gpus"] = [torch.cuda.get_device_name(g) for g in range(G)]
TM["peak_gb"] = round(max(torch.cuda.max_memory_allocated(g) for g in range(G)) / 2 ** 30, 2)
TM["process_wall_s"] = round(time.time() - T0, 2)
json.dump(dict(times=TM, lists=LISTS, fuse=fuse), open(outp, "w"))
print(json.dumps(TM))
