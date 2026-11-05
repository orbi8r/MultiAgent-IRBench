"""Qwen3 Reranker yes/no scorer with the same prompt, 480 token body cap and length sorted batches as gpu_score.py and score_pool.py.
The score cache holds {qid: {piece_id: log p(yes)}}, one file per run, and survives retries and restarts. The files are small.
Modes: fp16 (all on the card), nf4 (bitsandbytes 4 bit, the 8B on an 11 GB card), off (fp16 with CPU offload, GPUMEM env)."""
import json, os, time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from study_helpers import INSTR, atomic, log

PRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
       "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Scorer:
    def __init__(self, base, adapter=None, mode="fp16", maxb=480, bs=None):
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(base, padding_side="left")
        if mode == "nf4":
            from transformers import BitsAndBytesConfig
            m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
        elif mode == "off":
            m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto",
                                                     max_memory={0: os.environ.get("GPUMEM", "9GiB"), "cpu": "24GiB"})
        else:
            m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16).cuda()
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.m = m.eval()
        self.dev = self.m.device if mode != "off" else torch.device("cuda:0")
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.pre, self.suf = self.tok.encode(PRE, add_special_tokens=False), self.tok.encode(SUF, add_special_tokens=False)
        self.maxb, self.mode = maxb, mode
        small = "0.6B" in base
        self.bs = max(1, (bs or (32 if small else 8 if mode == "nf4" else 12 if mode == "off" else 6)) // int(os.environ.get("BS_DIV", 1)))
        torch.cuda.synchronize(); self.load_s = time.time() - t0
        torch.cuda.reset_peak_memory_stats()
        self.pairs = 0; self.secs = 0.0

    def score(self, items, every=0):
        """items: [(query text, piece ctx)] -> [log p(yes)] in the same order."""
        t0 = time.time()
        bodies = [self.tok.encode(f"<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {c}", add_special_tokens=False)[:self.maxb] for q, c in items]
        idx = sorted(range(len(items)), key=lambda i: len(bodies[i]))
        out = [0.0] * len(items)
        with torch.no_grad():
            for n, s in enumerate(range(0, len(idx), self.bs)):
                ch = idx[s:s + self.bs]
                b = self.tok.pad({"input_ids": [self.pre + bodies[i] + self.suf for i in ch]}, padding=True, return_tensors="pt").to(self.dev)
                lg = self.m(**b, logits_to_keep=1).logits[:, -1, :]
                lp = torch.log_softmax(torch.stack([lg[:, self.no], lg[:, self.yes]], 1).float(), 1)[:, 1]
                for i, v in zip(ch, lp.tolist()):
                    out[i] = v
                if every and n % every == 0:
                    log(f"  scored {s}/{len(idx)}")
        torch.cuda.synchronize()
        self.pairs += len(items); self.secs += time.time() - t0
        return out

    def peak(self):
        return torch.cuda.max_memory_allocated() / 2 ** 30


class Cache:
    """{qid: {piece: score}} of one reader on one run, persisted as json."""
    def __init__(self, path, seeds=()):
        self.path = Path(path)
        self.d = json.load(open(self.path)) if self.path.exists() else {}
        for s in seeds:
            if os.path.exists(s):
                for qid, v in json.load(open(s)).items():
                    for x, sc in v.items():
                        self.d.setdefault(qid, {}).setdefault(x, sc)

    def missing(self, wanted):
        return [(qid, x) for qid, ids in wanted.items() for x in dict.fromkeys(ids) if x not in self.d.get(qid, {})]

    def fill(self, scorer, wanted, qtext, ctx, every=400):
        """Score every wanted pair not cached yet (wanted: {qid: [piece ids]}), save, return the number scored."""
        miss = self.missing(wanted)
        for s in range(0, len(miss), 3000):  # saved every 3000 pairs so a killed job keeps its progress
            part = miss[s:s + 3000]
            vals = scorer.score([(qtext[qid], ctx[x]) for qid, x in part], every=every)
            for (qid, x), v in zip(part, vals):
                self.d.setdefault(qid, {})[x] = v
            self.save()
        return len(miss)

    def save(self):
        atomic(self.path, self.d)
