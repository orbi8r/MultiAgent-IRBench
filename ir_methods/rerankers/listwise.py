"""Zero shot listwise reader: a local instruct model reads the question and the top N reranked pieces (with their context
head) and names the pieces that state the answer themselves, best first. No training on any benchmark data.
    python listwise.py RUN BASE_TAG BASE_MODEL LLM N OUT_TAG
Writes data/RUN/list_OUT_TAG.json = {qid: {"order": [ids of the top N in base order], "picks": [indices], "raw": str}}."""
import json, os, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

run, btag, bmodel, llm, N, otag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
D = f"data/{run}"
G = json.load(open(f"{D}/gpu_{btag}.json"))
P = {}
for l in open(f"{D}/pieces.jsonl"):
    p = json.loads(l); P[p["id"]] = p
Q = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/questions.jsonl")}
SYS = ("You help a team of AI agents find things in the saved record of their own work. You are shown a question and "
       "numbered pieces of the record. A piece counts only if its own text states the asked fact (the value, name, path, "
       "decision or reason itself), not if it only mentions the topic, plans to find it, or asks for it.")


def prompt(q, ids):
    parts = []
    for i, x in enumerate(ids):
        p = P[x]
        head = p["ctx"].split("\n", 1)[0][:160]
        parts.append(f"[{i}] ({head})\n{p['text'][:700]}")
    return (f"Question (asked by {q['asker']}): {q['question']}\n\n" + "\n\n".join(parts) +
            "\n\nWhich pieces state the answer themselves? Reply with their numbers only, best first, comma separated, "
            "or 'none'.")


t0 = time.time()
tok = AutoTokenizer.from_pretrained(llm, padding_side="left")
multi = torch.cuda.device_count() > 1
if os.environ.get("OFFLOAD") == "1":  # one 11 GB card only: the rest of the weights wait in CPU memory (same fp16 math)
    model = AutoModelForCausalLM.from_pretrained(llm, dtype=torch.float16, device_map="auto", max_memory={0: os.environ.get("GPUMEM", "9GiB"), "cpu": "24GiB"})
else:
    model = AutoModelForCausalLM.from_pretrained(llm, dtype=torch.float16, device_map="auto" if multi else None)
    if not multi:
        model = model.cuda()
model.eval()
load = time.time() - t0; t0 = time.time()
items = []
for qid, c in G["cands"].items():
    order = [x for _, x in sorted(zip(G["scores"][bmodel][qid], c), key=lambda t: -t[0])][:N]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt(Q[qid], order)}]
    kw = {"enable_thinking": False} if "Qwen3" in llm and "Instruct" not in llm else {}
    items.append((qid, order, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)))
items.sort(key=lambda t: len(t[2]))
out = {}
bs = max(1, int(os.environ.get("BS", 4)) // int(os.environ.get("BS_DIV", 1)))
with torch.no_grad():
    for s in range(0, len(items), bs):
        ch = items[s:s + bs]
        enc = tok([t[2] for t in ch], return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        for (qid, order, _), row in zip(ch, g[:, enc["input_ids"].shape[1]:]):
            raw = tok.decode(row, skip_special_tokens=True)
            picks = [int(m) for m in re.findall(r"\d+", raw) if int(m) < len(order)]
            out[qid] = dict(order=order, picks=list(dict.fromkeys(picks)), raw=raw)
json.dump(out, open(f"{D}/list_{otag}.json", "w"))
print("DONE", run, otag, f"load {load:.0f}s read {time.time() - t0:.0f}s questions {len(out)}")
