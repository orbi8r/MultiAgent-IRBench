"""Scores the fuse0 top 100 of every question with a saved student cross encoder (for example rstul6 or rstul12), max length 512, batch 128.
Each pair is the question text with its asker, and the piece context. The models are frozen and nothing is trained here.
    python score_student.py RUN MODELDIR TAG -> data/<run>/<TAG>sub1.json {qid: {piece: logit}}  (needs data/<run>/f1.json)"""
import json, os, sys, time
if os.environ.get("NO_TRITON"):
    sys.modules["triton"] = None
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

run, mdir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
D = f"data/{run}"
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/pieces.jsonl")}
Q = [json.loads(l) for l in open(f"{D}/questions.jsonl")]
L = json.load(open(f"{D}/f1.json"))["fuse0"]
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForSequenceClassification.from_pretrained(mdir).half().cuda().eval()
qt = {q["qid"]: f"{q['question']} (asked by {q['asker']})" for q in Q}
pairs = [(q["qid"], x) for q in Q for x in L[q["qid"]][:100]]
idx = sorted(range(len(pairs)), key=lambda i: len(P[pairs[i][1]]["ctx"]))
out = [0.0] * len(pairs)
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(idx), 128):
        ch = idx[s:s + 128]
        enc = tok([qt[pairs[i][0]] for i in ch], [P[pairs[i][1]]["ctx"] for i in ch], truncation="only_second", max_length=512,
                  padding=True, return_tensors="pt").to("cuda")
        for i, v in zip(ch, model(**enc).logits[:, 0].float().tolist()):
            out[i] = v
torch.cuda.synchronize()
res = {}
for (a, b), v in zip(pairs, out):
    res.setdefault(a, {})[b] = v
json.dump(res, open(f"{D}/{tag}sub1.json", "w"))
print("DONE", run, tag, len(pairs), "pairs", round(time.time() - t0, 1), "s", torch.cuda.get_device_name(0))
