"""Document expansion for every piece with an off the shelf local model (no keys, no benchmark questions, no tuning).
    python doc2query_generate.py RUN t5    -> data/<run>/d2q_t5.jsonl    castorini/doc2query-t5-base-msmarco, 5 sampled queries
    python doc2query_generate.py RUN qwen  -> data/<run>/d2q_qwen.jsonl  Qwen/Qwen3-1.7B, greedy, asks for two questions per piece
Each line {id, q: [...]}; pieces with fewer than two words of three or more letters are skipped."""
import json, os, re, sys, time
if os.environ.get("NO_TRITON"):  # set on machines without Python.h, where triton cannot build its helper; generate works without it
    sys.modules["triton"] = None
import torch

run, mode = sys.argv[1], sys.argv[2]
P = [json.loads(l) for l in open(f"data/{run}/pieces.jsonl")]
P = [p for p in P if len(re.findall(r"[A-Za-z]{3,}", p["text"])) >= 2]
torch.manual_seed(0)
t0 = time.time()
out = open(f"data/{run}/d2q_{mode}.jsonl", "w")
if mode == "t5":
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    name = "castorini/doc2query-t5-base-msmarco"
    tok = T5Tokenizer.from_pretrained(name); m = T5ForConditionalGeneration.from_pretrained(name).half().cuda().eval()
    order = sorted(range(len(P)), key=lambda i: len(P[i]["ctx"]))
    for s in range(0, len(order), 64):
        ch = order[s:s + 64]
        enc = tok([P[i]["ctx"][:1500] for i in ch], return_tensors="pt", padding=True, truncation=True, max_length=384).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=48, do_sample=True, top_k=10, num_return_sequences=5, disable_compile=True)
        dec = tok.batch_decode(g, skip_special_tokens=True)
        for k, i in enumerate(ch):
            out.write(json.dumps({"id": P[i]["id"], "q": dec[5 * k:5 * k + 5]}) + "\n")
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    def prompt(p):
        return ("Here is one piece of the saved work record of a team of AI agents, with its context.\n---\n" + p["ctx"][:1200] +
                "\n---\nWrite two different questions a teammate might later ask whose answers are specific facts stated in this "
                "piece (a number, name, path, command, decision, reason, status or date). Use a teammate's own words, not phrases "
                "copied from the piece. One question per line, nothing else.")
    texts = [tok.apply_chat_template([{"role": "user", "content": prompt(p)}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in P]
    order = sorted(range(len(P)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), 24):
        ch = order[s:s + 24]
        enc = tok([texts[i] for i in ch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=70, do_sample=False, pad_token_id=tok.pad_token_id, disable_compile=True)
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for k, i in enumerate(ch):
            out.write(json.dumps({"id": P[i]["id"], "q": [x.strip() for x in dec[k].split("\n") if x.strip()][:3]}) + "\n")
        if s % 2400 == 0:
            print(s, f"{time.time()-t0:.0f}s", flush=True); out.flush()
out.close()
print("DONE", run, mode, len(P), f"{time.time()-t0:.0f}s")
