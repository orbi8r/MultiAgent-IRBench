"""Rescores the top N of a saved first reranker ranking with a larger Qwen3 Reranker.
    python cascade.py RUN BASE_TAG BASE_MODEL BIG_MODEL N OUT_TAG
Reads the base ranking from data/RUN/gpu_BASE_TAG.json and writes data/RUN/gpu_OUT_TAG.json. The candidate lists keep the base order.
scores[BIG_MODEL] holds the big model's log p(yes) for the top N in that order, with lower values after them that keep the base order for the rest."""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

run, btag, bmodel, big, N, otag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
D = f"data/{run}"
G = json.load(open(f"{D}/gpu_{btag}.json"))
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/pieces.jsonl")}
Q = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/questions.jsonl")}
INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(big, padding_side="left")
multi = torch.cuda.device_count() > 1
if os.environ.get("OFFLOAD") == "1":  # one 11 GB card only: the rest of the weights wait in CPU memory (same fp16 math)
    model = AutoModelForCausalLM.from_pretrained(big, dtype=torch.float16, device_map="auto", max_memory={0: os.environ.get("GPUMEM", "9GiB"), "cpu": "24GiB"})
else:
    model = AutoModelForCausalLM.from_pretrained(big, dtype=torch.float16, device_map="auto" if multi else None)
    if not multi:
        model = model.cuda()
model.eval()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
pre = tok.encode("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                 "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
load = time.time() - t0; t0 = time.time()
cands, scores = {}, {}
items = []
for qid, c in G["cands"].items():
    base = [x for _, x in sorted(zip(G["scores"][bmodel][qid], c), key=lambda t: -t[0])]
    cands[qid] = base
    q = Q[qid]
    for x in base[:N]:
        items.append((qid, x, tok.encode(f"<Instruct>: {INSTR}\n<Query>: {q['question']} (asked by {q['asker']})\n<Document>: {P[x]['ctx']}",
                                         add_special_tokens=False)[:480]))
order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
out = {}
bs = max(1, int(os.environ.get("BS", 4)) // int(os.environ.get("BS_DIV", 1)))
with torch.no_grad():
    for s in range(0, len(order), bs):
        ch = order[s:s + bs]
        batch = tok.pad({"input_ids": [pre + items[i][2] + suf for i in ch]}, padding=True, return_tensors="pt").to(model.device)
        lg = model(**batch, logits_to_keep=1).logits[:, -1, :]
        lp = torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], 1).float(), 1)[:, 1]
        for i, v in zip(ch, lp.tolist()):
            out[(items[i][0], items[i][1])] = v
for qid, base in cands.items():
    top = [out[(qid, x)] for x in base[:N]]
    lo = min(top) - 1
    scores[qid] = top + [lo - 1e-3 * i for i in range(len(base) - N)]
json.dump(dict(cands=cands, scores={big: scores}, times=dict(load=load, score=time.time() - t0)), open(f"{D}/gpu_{otag}.json", "w"))
print("DONE", run, otag, f"load {load:.0f}s score {time.time() - t0:.0f}s pairs {len(items)}")
