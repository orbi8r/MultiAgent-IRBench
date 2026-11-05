"""Teacher scores on public training rows (never benchmark data). A public reader scores the positive and the first D-1
negatives of every row in shard SHARD of NSHARDS. Writes OUT as jsonl, one {file, line, scores} record per row.
    python teacher_scores.py BASE LORA OUT SHARD NSHARDS D FILE [FILE ...]"""
import json, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base, lora, out, sh, nsh, D = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")
tok = AutoTokenizer.from_pretrained(base, padding_side="left")
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16).cuda()
m = PeftModel.from_pretrained(m, lora).merge_and_unload().eval()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
pre = tok.encode("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                 "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
items, k = [], 0
for f in sys.argv[7:]:
    for ln, l in enumerate(open(f)):
        k += 1
        if k % nsh != sh:
            continue
        r = json.loads(l)
        for j, d in enumerate([r["pos"]] + r["negs"][:D - 1]):
            items.append((f, ln, j, tok.encode(f"<Instruct>: {INSTR}\n<Query>: {r['query']}\n<Document>: {d}", add_special_tokens=False)[:480]))
order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
sc = [0.0] * len(items)
bs = 32 if "0.6B" in base else 8
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(order), bs):
        ch = order[s:s + bs]
        b = tok.pad({"input_ids": [pre + items[i][3] + suf for i in ch]}, padding=True, return_tensors="pt").to("cuda")
        lg = m(**b, logits_to_keep=1).logits[:, -1, :]
        lp = (lg[:, yes] - lg[:, no]).float().tolist()
        for i, v in zip(ch, lp):
            sc[i] = v
res = {}
for (f, ln, j, _), v in zip(items, sc):
    res.setdefault((f, ln), {})[j] = v
with open(out, "w") as fo:
    for (f, ln), d in res.items():
        fo.write(json.dumps(dict(file=f, line=ln, scores=[d[j] for j in sorted(d)])) + "\n")
print("DONE", out, len(items), "pairs", f"{time.time() - t0:.0f}s")
