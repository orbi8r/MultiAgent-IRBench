"""Small student reader trained on public agent traces only, never on benchmark data.
The rows are {query, pos, negs} from the ROWS_GLOB files. They are local Qwen3 4B questions over public SWE agent trajectories (pub_*/train_synth.jsonl).
The base is an off the shelf MS MARCO cross encoder, trained for 1 epoch with a listwise softmax over 1 positive and 7 negatives.
Then it scores the benchmark pools (pool1, pool2) for offline evaluation, and times serving over fuse0 top 20, 50 and 100.
    python train_student.py BASE OUTTAG ROWS_GLOB"""
import json, glob, sys, time, random, os
if os.environ.get("NO_TRITON"):  # set on machines without Python.h, where triton cannot build
    sys.modules["triton"] = None
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

base, tag, pat = sys.argv[1], sys.argv[2], sys.argv[3]
MAXLEN = int(os.environ.get("MAXLEN", 256))
rows = [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f)]
random.Random(0).shuffle(rows)
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("LR", 3e-5)))
G, B = 8, 8
steps = len(rows) // B
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * max(0.0, 1 - s / steps))
scaler = torch.cuda.amp.GradScaler()
model.train(); t0 = time.time()
for s in range(steps):
    batch = rows[s * B:(s + 1) * B]
    qs, ds = [], []
    for r in batch:
        negs = r["negs"][:G - 1]
        while len(negs) < G - 1:
            negs.append(negs[-1] if negs else "")
        for d in [r["pos"]] + negs:
            qs.append(r["query"]); ds.append(d)
    enc = tok(qs, ds, truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(**enc).logits.view(len(batch), G)
    loss = torch.nn.functional.cross_entropy(logits.float(), torch.zeros(len(batch), dtype=torch.long, device="cuda"))
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    if s % 500 == 0:
        print("step", s, steps, round(loss.item(), 3), f"{time.time()-t0:.0f}s", flush=True)
model.eval().half()
os.makedirs(f"models/{tag}", exist_ok=True); model.save_pretrained(f"models/{tag}"); tok.save_pretrained(f"models/{tag}")
print("trained", len(rows), "rows", f"{time.time()-t0:.0f}s", flush=True)


def score(pairs, bs=128):
    out = [0.0] * len(pairs)
    idx = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            ch = idx[s:s + bs]
            enc = tok([pairs[i][0] for i in ch], [pairs[i][1] for i in ch], truncation="only_second", max_length=MAXLEN,
                      padding=True, return_tensors="pt").to("cuda")
            for i, v in zip(ch, model(**enc).logits[:, 0].float().tolist()):
                out[i] = v
    return out


for run in ("online_shop", "library_maintainers", "literature_review"):
    D = f"data/{run}"
    P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/pieces.jsonl")}
    Q = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/questions.jsonl")}
    qt = {k: f"{q['question']} (asked by {q['asker']})" for k, q in Q.items()}
    for rnd in (1, 2):
        pool = json.load(open(f"{D}/pool{rnd}.json"))
        pairs = [(qid, x) for qid, xs in pool.items() for x in xs]
        v = score([(qt[a], P[b]["ctx"]) for a, b in pairs])
        res = {}
        for (a, b), s in zip(pairs, v):
            res.setdefault(a, {})[b] = s
        json.dump(res, open(f"{D}/{tag}sub{rnd}.json", "w"))
    L = json.load(open(f"{D}/fuse0.json")); T = {}
    for K in (20, 50, 100):
        pairs = [(qt[qid], P[x]["ctx"]) for qid in Q for x in L[qid][:K]]
        torch.cuda.synchronize(); t = time.time(); score(pairs); torch.cuda.synchronize()
        T[f"K{K}_s"] = round(time.time() - t, 2); T[f"K{K}_pairs"] = len(pairs)
    print("TIME", run, T, torch.cuda.get_device_name(0), flush=True)
