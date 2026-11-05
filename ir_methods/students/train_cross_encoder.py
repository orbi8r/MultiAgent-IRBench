"""Small cross encoder student (BERT class, e.g. MiniLM) trained on public pseudo record rows only, with a listwise softmax over
1 positive and G-1 hard negatives. If a row carries "teacher" scores (positive first, then negatives, from a public teacher on
public records), the loss is the KL to the teacher's softened distribution instead.
    python train_cross_encoder.py BASE OUTDIR EPOCHS MAXLEN FILE [FILE ...]"""
import json, random, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

base, outdir, epochs, maxlen = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(l) for f in sys.argv[5:] for l in open(f)]
import os
G, LR, BQ, TEMP = 8, 3e-5, int(os.environ.get("BQ", 4)), 2.0
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1, ignore_mismatched_sizes=True).cuda()
if os.environ.get("GC") == "1":
    model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = int(len(rows) * epochs / BQ)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / max(1, steps)))
rnd = random.Random(0)
scaler = torch.amp.GradScaler()
model.train()
t0, run = time.time(), 0.0
order = []
for step in range(steps):
    if len(order) < BQ:
        order = list(range(len(rows))); rnd.shuffle(order)
    batch = [rows[order.pop()] for _ in range(BQ)]
    qs, ds, tgt = [], [], []
    for r in batch:
        idx = list(range(len(r["negs"]))); rnd.shuffle(idx); idx = idx[:G - 1]
        docs = [r["pos"]] + [r["negs"][i] for i in idx]
        qs += [r["query"]] * len(docs); ds += docs
        if "teacher" in r:
            t = torch.tensor([r["teacher"][0]] + [r["teacher"][1 + i] for i in idx]) / TEMP
            tgt.append(torch.softmax(t, 0))
        else:
            tgt.append(torch.nn.functional.one_hot(torch.tensor(0), len(docs)).float())
    enc = tok(qs, ds, truncation="only_second", max_length=maxlen, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        s = model(**enc).logits.view(BQ, -1).float()
    loss = -(torch.stack(tgt).cuda() * torch.log_softmax(s, 1)).sum(1).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
    run += loss.item()
    if (step + 1) % 200 == 0:
        print(f"step {step + 1}/{steps} loss {run / 200:.3f} {time.time() - t0:.0f}s", flush=True); run = 0.0
model.save_pretrained(outdir); tok.save_pretrained(outdir)
print("DONE", outdir, f"{time.time() - t0:.0f}s")
