"""GPU stage: dense first stage over contextual pieces, then score the fused candidates with rerankers.
Writes data/<run>/gpu_<tag>.json = {dense: {qid: [ids]}, cands: {qid: [ids]}, scores: {model: {qid: [float]}}, times: {...}}.
    python gpu_score.py RUN TAG EMBEDDER RERANKER[,RERANKER...] [K]
"""
import json, os, sys, time
from collections import defaultdict
import torch

run, tag, emb_name, rr_names = sys.argv[1:5]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 200
D = f"data/{run}"
P = [json.loads(l) for l in open(f"{D}/pieces.jsonl")]
Q = [json.loads(l) for l in open(f"{D}/questions.jsonl")][:int(os.environ.get("NQ", 10**9))]  # NQ caps the question count, for smoke runs only
lex = json.load(open(f"{D}/lex.json"))
ctx = {p["id"]: p["ctx"] for p in P}
CTXF = os.environ.get("CTXFILE")  # JSON map of piece id to prefixed context; it replaces the document text, candidates and prompt stay the same
if CTXF:
    ctx = json.load(open(CTXF, encoding="utf-8"))
INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")
QI = {"location": "Given a question asking where a file or message is kept, judge whether this piece holds the content "
                  "of that file or message, as it was written or read",
      "who": "Given a question asking who did something and in which shift, judge whether this piece is the step where "
             "that action itself was done (the command, edit, commit or message), not a later mention of it",
      "history": "Given a question about what happened earlier in an AI agent team's work, judge whether this piece is "
                 "the original step where it happened, not a later mention of it"}
import os
USE_QI = os.environ.get("QINSTR") == "1"
MAXB = int(os.environ.get("MAXB", 480))  # tokens of instruction + query + document kept per pair
times = {}


def qtext(q):
    return f"{q['question']} (asked by {q['asker']})"


def rrf(lists, k=60):
    s = defaultdict(float)
    for l in lists:
        for r, x in enumerate(l):
            s[x] += 1 / (k + r)
    return sorted(s, key=lambda x: -s[x])


t0 = time.time()
CF = os.environ.get("CANDS_FROM")  # reuse the frozen candidate and dense lists of an earlier output file
if CF:
    _g = json.load(open(CF)); dense = _g["dense"]; times["cands_from"] = CF
elif emb_name == "none":
    dense = {q["qid"]: [] for q in Q}
else:
  from sentence_transformers import SentenceTransformer
  emb = SentenceTransformer(emb_name, device="cuda", model_kwargs={"torch_dtype": torch.float16})
  emb.max_seq_length = 384
  torch.cuda.synchronize(); times["load_embedder"] = time.time() - t0
  t0 = time.time()
  docs = [p["ctx"] for p in P]
  order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
  E = emb.encode([docs[i] for i in order], batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
  E = E[torch.argsort(torch.tensor(order, device=E.device))]
  qkw = {"prompt_name": "query"} if "query" in getattr(emb, "prompts", {}) else {}
  QE = emb.encode([qtext(q) for q in Q], batch_size=64, convert_to_tensor=True, normalize_embeddings=True, **qkw)
  sim = QE @ E.T
  top = torch.topk(sim, 300, dim=1).indices.tolist()
  dense = {q["qid"]: [P[i]["id"] for i in top[j]] for j, q in enumerate(Q)}
  torch.cuda.synchronize(); times["dense"] = time.time() - t0
  del emb, E; torch.cuda.empty_cache()
cands = {q["qid"]: _g["cands"][q["qid"]] for q in Q} if CF else {q["qid"]: rrf([lex[q["qid"]]["ctx"], lex[q["qid"]]["plain"], dense[q["qid"]]])[:K] for q in Q}
pairs = [(j, x) for j, q in enumerate(Q) for x in cands[q["qid"]]]

scores = {}
for name in ([] if rr_names == "none" else rr_names.split(",")):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = [0.0] * len(pairs)
    if "Qwen3-Reranker" in name:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        bname, _, lora = name.partition("+lora=")
        tok = AutoTokenizer.from_pretrained(bname, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(bname, torch_dtype=torch.float16).cuda().eval()
        if lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora).merge_and_unload().eval()
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
               "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
        suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        pre_ids, suf_ids = tok.encode(pre, add_special_tokens=False), tok.encode(suf, add_special_tokens=False)
        torch.cuda.synchronize(); times[f"load_{name}"] = time.time() - t0; t0 = time.time()
        bodies = [tok.encode(f"<Instruct>: {QI.get(Q[j]['qtype'], INSTR) if USE_QI else INSTR}\n<Query>: {qtext(Q[j])}\n<Document>: {ctx[x]}", add_special_tokens=False)[:MAXB]
                  for j, x in pairs]
        idx = sorted(range(len(pairs)), key=lambda i: len(bodies[i]))
        bs = max(1, (32 if "0.6B" in name else 6) // int(os.environ.get("BS_DIV", 1)))
        with torch.no_grad():
            for s in range(0, len(idx), bs):
                chunk = idx[s:s + bs]
                batch = tok.pad({"input_ids": [pre_ids + bodies[i] + suf_ids for i in chunk]}, padding=True, return_tensors="pt").to("cuda")
                logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
                lp = torch.log_softmax(torch.stack([logits[:, no], logits[:, yes]], 1).float(), 1)[:, 1]
                for i, v in zip(chunk, lp.tolist()):
                    out[i] = v
    else:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(name, device="cuda", max_length=min(512, MAXB + 32), model_kwargs={"torch_dtype": torch.float16})
        torch.cuda.synchronize(); times[f"load_{name}"] = time.time() - t0; t0 = time.time()
        out = ce.predict([(qtext(Q[j]), ctx[x]) for j, x in pairs], batch_size=64).tolist()
        model = ce
    torch.cuda.synchronize(); times[f"peak_gb_{name}"] = torch.cuda.max_memory_allocated() / 2**30; times[f"score_{name}"] = time.time() - t0; times[f"pairs_per_s_{name}"] = len(pairs) / times[f"score_{name}"]
    sc = defaultdict(list)
    for (j, x), v in zip(pairs, out):
        sc[Q[j]["qid"]].append(v)
    scores[name] = dict(sc)
    del model; torch.cuda.empty_cache()
    print(name, times, flush=True)
    json.dump(dict(dense=dense, cands=cands, scores=scores, times=times, gpu=torch.cuda.get_device_name(0)), open(f"{D}/gpu_{tag}.json", "w"))
json.dump(dict(dense=dense, cands=cands, scores=scores, times=times, gpu=torch.cuda.get_device_name(0)), open(f"{D}/gpu_{tag}.json", "w"))
print("DONE", run, tag, times)
