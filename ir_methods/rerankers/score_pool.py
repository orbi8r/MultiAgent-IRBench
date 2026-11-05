"""Score a pool of pieces with a Qwen3 Reranker 0.6B LoRA reader, using the standard reranker prompt (batch 32, 480 token body cap).
    POOLF=data/<run>/pool1.json OUTF=data/<run>/pubsub1 python score_pool.py RUN ADAPTER SHARD NSHARD
Writes OUTF.sh<SHARD>.json {qid: {piece: log p(yes)}} (questions with index % NSHARD == SHARD). Merge with merge_shards.py."""
import json, os, sys, time
if os.environ.get("NO_TRITON"):
    sys.modules["triton"] = None
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

run, adapter, sh, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
D = f"data/{run}"
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/pieces.jsonl")}
Q = [q for j, q in enumerate(json.loads(l) for l in open(f"{D}/questions.jsonl")) if j % nsh == sh]
POOL = json.load(open(os.environ["POOLF"]))
INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")
t0 = time.time()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16).cuda().eval()
model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
pre = tok.encode("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                 "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
print("load", round(time.time() - t0, 1), flush=True)
pairs = [(q, x) for q in Q for x in POOL.get(q["qid"], [])]
heads = {q["qid"]: f"<Instruct>: {INSTR}\n<Query>: {q['question']} (asked by {q['asker']})\n<Document>: " for q in Q}
bodies = [tok.encode(heads[q["qid"]] + P[x]["ctx"], add_special_tokens=False)[:480] for q, x in pairs]
idx = sorted(range(len(pairs)), key=lambda i: len(bodies[i]))
vals = [0.0] * len(pairs)
with torch.no_grad():
    B = max(1, 32 // int(os.environ.get("BS_DIV", 1)))  # the job runner halves BS_DIV after a CUDA OOM
    for s in range(0, len(idx), B):
        chunk = idx[s:s + B]
        batch = tok.pad({"input_ids": [pre + bodies[i] + suf for i in chunk]}, padding=True, return_tensors="pt").to("cuda")
        logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
        lp = torch.log_softmax(torch.stack([logits[:, no], logits[:, yes]], 1).float(), 1)[:, 1]
        for i, v in zip(chunk, lp.tolist()):
            vals[i] = v
        if (s // B) % 100 == 0:
            print(s, len(idx), round(time.time() - t0), flush=True)
res = {}
for (q, x), v in zip(pairs, vals):
    res.setdefault(q["qid"], {})[x] = v
json.dump(res, open(f"{os.environ['OUTF']}.sh{sh}.json", "w"))
print("DONE", run, len(pairs), "pairs", round(time.time() - t0), "s", torch.cuda.get_device_name(0))
