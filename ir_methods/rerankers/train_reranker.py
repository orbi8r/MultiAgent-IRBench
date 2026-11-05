"""LoRA fine tune of a Qwen3 Reranker (yes/no head) with a listwise softmax over 1 positive and G-1 hard negatives.
    python train_reranker.py BASE OUTDIR EPOCHS TRAIN.jsonl [TRAIN2.jsonl ...]
Each line: {query, pos, negs: [...]} with texts already in the contextual form used at scoring time.
The training recipe is unchanged. Checkpointing is off unless the env sets CKPT and CKEVERY:
  CKPT=<file> CKEVERY=<steps>  save LoRA, optimizer, scheduler, scaler and RNG every CKEVERY steps (rounded up to a multiple of ACC)
                               and resume from the file when it exists (the row and negative order is replayed from the same seed).
The adapter is written to OUTDIR.tmp and renamed, so an existing OUTDIR is always complete."""
import json, os, sys, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base, outdir, epochs = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = [json.loads(l) for f in sys.argv[4:] for l in open(f)]
G, MAXLEN, LR, ACC = 8, 448, 1e-4, 4
INSTR = ("Given a question a team member asks about their AI agent team's saved work record, "
         "judge whether this piece of the record itself states the fact the question asks for")
pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
       "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
tok = AutoTokenizer.from_pretrained(base, padding_side="left")
big = True  # fp16 base with fp32 LoRA and a grad scaler on cards without bf16
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else (torch.float16 if big else torch.float32)
scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).cuda()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
SEED = os.environ.get("SEED")  # unset keeps the original unseeded LoRA init and shuffle seed 0
if SEED is not None:
    torch.manual_seed(int(SEED))
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
pre_ids, suf_ids = tok.encode(pre, add_special_tokens=False), tok.encode(suf, add_special_tokens=False)


def enc(q, d):
    return pre_ids + tok.encode(f"<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {d}", add_special_tokens=False)[:MAXLEN] + suf_ids


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
rnd = random.Random(int(SEED) if SEED is not None else 0)
steps = int(len(rows) * epochs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / max(1, steps / ACC)))
model.train()
CKPT, CKEVERY = os.environ.get("CKPT"), int(os.environ.get("CKEVERY", 0))
CKEVERY = -(-CKEVERY // ACC) * ACC if CKEVERY else 0
trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
step0 = 0
if CKPT and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    step0 = ck["step"]
    with torch.no_grad():
        for n, p in trainable.items():
            p.copy_(ck["lora"][n])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    torch.set_rng_state(ck["rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
    for s0 in range(step0):  # replay the row shuffle and the negative shuffles of the finished steps
        if s0 % len(rows) == 0:
            rnd.shuffle(rows)
        rnd.shuffle(rows[s0 % len(rows)]["negs"][:])
    print(f"resumed at step {step0}/{steps}", flush=True)
t0, run_loss = time.time(), 0.0
for step in range(step0, steps):
    if step % len(rows) == 0:
        rnd.shuffle(rows)
    r = rows[step % len(rows)]
    negs = r["negs"][:]
    rnd.shuffle(negs)
    docs = [r["pos"]] + negs[:G - 1]
    batch = tok.pad({"input_ids": [enc(r["query"], d) for d in docs]}, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
    s = (logits[:, yes] - logits[:, no]).float()
    loss = torch.nn.functional.cross_entropy(s.unsqueeze(0), torch.zeros(1, dtype=torch.long, device="cuda")) / ACC
    scaler.scale(loss).backward()
    run_loss += loss.item() * ACC
    if (step + 1) % ACC == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
    if (step + 1) % 200 == 0:
        print(f"step {step + 1}/{steps} loss {run_loss / 200:.3f} {time.time() - t0:.0f}s", flush=True)
        run_loss = 0.0
    if CKPT and CKEVERY and (step + 1) % CKEVERY == 0 and step + 1 < steps:
        torch.save(dict(step=step + 1, lora={n: p.detach().cpu() for n, p in trainable.items()}, opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)
        print(f"checkpoint at step {step + 1}", flush=True)
import shutil
shutil.rmtree(outdir + ".tmp", ignore_errors=True)
model.save_pretrained(outdir + ".tmp")
os.replace(outdir + ".tmp", outdir)
if CKPT and os.path.exists(CKPT):
    os.remove(CKPT)
print("DONE", outdir, f"{time.time() - t0:.0f}s")
