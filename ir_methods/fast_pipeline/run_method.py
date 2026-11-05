"""AgentTrace retrieval, end to end for one run, timed. Reads only the run's public files (blocks, events, questions).
    python run_method.py RUN_DIR OUT.json [--k 100] [--model Qwen/Qwen3-Reranker-0.6B] [--lora PATH]
1. Contextual pieces: each block with who wrote it, the step kind, the tool call that produced it and its event head.
2. Lexical first stage: BM25 over contextual text and over plain text (split identifiers, stemmed), fused by RRF, top K.
3. Reranker: Qwen3 Reranker scores (question, contextual piece) on every visible GPU in parallel.
4. List: reranked, exact text repeats dropped; location questions put the place the described file ended up first
   (file provenance: content -> file -> later shell moves -> cheapest pieces stating that path).
"""
import json, os, sys, time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contextual_pieces as C
from file_provenance import Files

INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")


@lru_cache(maxsize=None)
def stem(w):
    return C._stem.stemWord(w)


def toks(text):
    import re
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [stem(w) for w in (w.lower() for w in re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)) if w not in C.STOP]


class SparseBM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.vocab = {}
        rows, cols, vals = [], [], []
        lens = np.zeros(len(docs))
        for i, d in enumerate(docs):
            c = Counter(d)
            lens[i] = max(1, sum(c.values()))
            for w, n in c.items():
                rows.append(i); cols.append(self.vocab.setdefault(w, len(self.vocab))); vals.append(n)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float32)
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        self.idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * lens / lens.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.W = sp.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def topk(self, q, k):
        ids = sorted({self.vocab[w] for w in q if w in self.vocab})
        if not ids:
            return []
        s = np.asarray(self.W[:, ids] @ self.idf[ids]).ravel()
        k = min(k, int((s > 0).sum()))
        top = np.argpartition(-s, k - 1)[:k] if k else []
        return sorted(top, key=lambda i: -s[i])


def rrf(lists, k=60):
    s = defaultdict(float)
    for l in lists:
        for r, x in enumerate(l):
            s[x] += 1 / (k + r)
    return sorted(s, key=lambda x: -s[x])


def score_worker(args):
    gpu, model, lora, items = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16).cuda().eval()
    if lora:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, lora).merge_and_unload().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok.encode("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                     "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    bodies = [tok.encode(f"<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {d}", add_special_tokens=False)[:480] for q, d in items]
    order = sorted(range(len(items)), key=lambda i: len(bodies[i]))
    out = [0.0] * len(items)
    with torch.no_grad():
        for s in range(0, len(order), 32):
            ch = order[s:s + 32]
            batch = tok.pad({"input_ids": [pre + bodies[i] + suf for i in ch]}, padding=True, return_tensors="pt").to("cuda")
            lg = m(**batch, logits_to_keep=1).logits[:, -1, :]
            lp = torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], 1).float(), 1)[:, 1]
            for i, v in zip(ch, lp.tolist()):
                out[i] = v
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--lora", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--dense", default="")
    a = ap.parse_args()
    T = {}
    t0 = time.time()
    d = Path(a.run)
    events = [json.loads(l) for l in open(d / "events.jsonl")]
    blocks = [json.loads(l) for l in open(d / "blocks.jsonl")]
    qs = [json.loads(l) for l in open(d / "questions.jsonl")]
    P = C.build(events, blocks)
    ids = [p["id"] for p in P]
    text = {p["id"]: p["text"] for p in P}
    T["pieces"] = time.time() - t0; t0 = time.time()
    bc, bp = SparseBM25([toks(p["ctx"]) for p in P]), SparseBM25([toks(p["text"]) for p in P])
    T["index"] = time.time() - t0; t0 = time.time()
    dense = {q["qid"]: [] for q in qs}
    if a.dense:
        import torch
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(a.dense, device="cuda:0", model_kwargs={"dtype": torch.float16})
        emb.max_seq_length = 256
        E = emb.encode([p["ctx"] for p in P], batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
        qkw = {"prompt_name": "query"} if "query" in getattr(emb, "prompts", {}) else {}
        QE = emb.encode([f"{q['question']} (asked by {q['asker']})" for q in qs], convert_to_tensor=True, normalize_embeddings=True, **qkw)
        top = torch.topk(QE @ E.T, 300, dim=1).indices.tolist()
        dense = {q["qid"]: top[j] for j, q in enumerate(qs)}
        del emb, E; torch.cuda.empty_cache()
    T["dense"] = time.time() - t0; t0 = time.time()
    cands = {}
    for q in qs:
        qt = toks(q["question"])
        cands[q["qid"]] = [ids[i] for i in rrf([bc.topk(qt, 300), bp.topk(qt, 300)] + ([dense[q["qid"]]] if a.dense else []))[:a.k]]
    T["first_stage"] = time.time() - t0; t0 = time.time()
    ctxt = {p["id"]: p["ctx"] for p in P}
    pairs = [(q["qid"], x) for q in qs for x in cands[q["qid"]]]
    items = [(f"{q['question']} (asked by {q['asker']})", ctxt[x]) for q in qs for x in cands[q["qid"]]]
    import torch
    n = max(1, torch.cuda.device_count())
    import multiprocessing as mp
    chunks = [list(range(i, len(items), n)) for i in range(n)]
    with mp.get_context("spawn").Pool(n) as pool:
        res = pool.map(score_worker, [(g, a.model, a.lora, [items[i] for i in chunks[g]]) for g in range(n)])
    sc = [0.0] * len(items)
    for g in range(n):
        for i, v in zip(chunks[g], res[g]):
            sc[i] = v
    T["rerank"] = time.time() - t0; t0 = time.time()
    S = defaultdict(dict)
    for (qid, x), v in zip(pairs, sc):
        S[qid][x] = v
    F = Files(events, P)
    F.by_event = defaultdict(list)
    for p in P:
        F.by_event[p["event"]].append(p["id"])
    import build_list
    sub = {q["qid"]: build_list.build(F, q, cands[q["qid"]], [S[q["qid"]][x] for x in cands[q["qid"]]], text) for q in qs}
    T["lists"] = time.time() - t0
    json.dump(sub, open(a.out, "w"))
    if a.dump:
        json.dump(dict(cands=cands, scores={q: [S[q][x] for x in cands[q]] for q in cands}), open(a.dump, "w"))
    print("TIMES", {k: round(v, 1) for k, v in T.items()}, "total", round(sum(T.values()), 1), "gpus", n, flush=True)


if __name__ == "__main__":
    main()
