"""Zero shot listwise reader (same prompt as listwise.py) over the top 10 of a base reader's ranking in data/<run>/gpu_z2s.json, with several tags sharing one model load.
    python listwise_multi.py LLM fp16|nf4 TAG|BASEKEY [TAG|BASEKEY ...]     BASEKEY: zs06 or ftpub (scores in data/<run>/gpu_z2s.json)
The model names the pieces that state the answer; they go first in that order, the rest keeps the base reader order, then the list builder.
A batch that runs out of GPU memory is redone one question at a time; a single question that still does not fit gets its pieces cut to 350 chars (counted in the log)."""
import json, os, re, sys, time
import torch
import study_helpers, tags

llm, quant = sys.argv[1:3]
specs = [s.split("|") for s in sys.argv[3:]]
ZS06 = "Qwen/Qwen3-Reranker-0.6B"
KEY = {"zs06": ZS06, "ftpub": ZS06 + "+lora=models/ftpub"}
N = 10
SYS = ("You help a team of AI agents find things in the saved record of their own work. You are shown a question and "
       "numbered pieces of the record. A piece counts only if its own text states the asked fact (the value, name, path, "
       "decision or reason itself), not if it only mentions the topic, plans to find it, or asks for it.")
frozen = {}
for tag, bk in specs:
    frozen[tag] = study_helpers.freeze(tag, dict(llm=llm, quant=quant, N=N, base=KEY[bk], gfile="gpu_z2s", max_new_tokens=24, piece_chars=700,
                                      thinking="off", builder="build_list.build SCAN 30 EARLIEST"))
    st = frozen[tag]
    tags.row(tag, "FIXED", f"{st['llm']} {st['quant']} listwise over the top {N} of {bk} on the z2 list, picks first then {bk} order; list builder (30, on): EARLIEST=1 LOC_SCAN=30 UTIL_A=0",
             "timing is the listwise read only, the base reader is not included")
specs = [(t, bk) for t, bk in specs]
M = {}


def model():
    if not M:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        M["tok"] = AutoTokenizer.from_pretrained(llm, padding_side="left")
        if quant == "nf4":
            from transformers import BitsAndBytesConfig
            M["m"] = AutoModelForCausalLM.from_pretrained(llm, torch_dtype=torch.float16, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
        else:
            M["m"] = AutoModelForCausalLM.from_pretrained(llm, torch_dtype=torch.float16).cuda().eval()
    return M["tok"], M["m"]


def prompt(q, P, ids, cap):
    parts = []
    for i, x in enumerate(ids):
        head = P[x]["ctx"].split("\n", 1)[0][:160]
        parts.append(f"[{i}] ({head})\n{P[x]['text'][:cap]}")
    return (f"Question (asked by {q['asker']}): {q['question']}\n\n" + "\n\n".join(parts) +
            "\n\nWhich pieces state the answer themselves? Reply with their numbers only, best first, comma separated, "
            "or 'none'.")


def gen(items, P, qd, tok, m, bs):
    """items: [(qid, order, cap)] -> {qid: raw text}"""
    def text(qid, order, cap):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt(qd[qid], P, order, cap)}]
        kw = {"enable_thinking": False} if "Qwen3" in llm and "Instruct" not in llm else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    out = {}
    items = sorted(items, key=lambda t: len(text(*t)))
    for s in range(0, len(items), bs):
        ch = items[s:s + bs]
        try:
            enc = tok([text(*t) for t in ch], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
            for t, row in zip(ch, g[:, enc["input_ids"].shape[1]:]):
                out[t[0]] = tok.decode(row, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            study_helpers.log("OOM, one by one", len(ch))
            for t in ch:
                for cap in (t[2], 350):
                    try:
                        enc = tok([text(t[0], t[1], cap)], return_tensors="pt", padding=True).to("cuda")
                        with torch.no_grad():
                            g = m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
                        out[t[0]] = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        if cap != t[2]:
                            study_helpers.log("cut to 350 chars", t[0])
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                else:
                    raise
    return out


for split, run in study_helpers.plan():
    todo = [s for s in specs if not study_helpers.have(s[0], split, run)]
    if not todo:
        continue
    G = json.load(open(f"data/{run}/gpu_z2s.json"))
    qs, F, rn, P = study_helpers.builder(run)
    Pd = {p["id"]: p for p in P}; qd = {q["qid"]: q for q in qs}
    for tag, bk in todo:
        lf = f"data/{run}/list_{tag}.json"
        if not os.path.exists(lf):
            tok, m = model()
            orders = {q["qid"]: study_helpers.ranked(G["cands"][q["qid"]], G["scores"][KEY[bk]][q["qid"]])[:N] for q in qs}
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            bs = max(1, int(os.environ.get("BS", 2)) // int(os.environ.get("BS_DIV", 1)))
            raw = gen([(k, v, 700) for k, v in orders.items()], Pd, qd, tok, m, bs)
            secs = time.time() - t0
            res = {}
            for qid, o in orders.items():
                picks = [int(x) for x in re.findall(r"\d+", raw[qid]) if int(x) < len(o)]
                res[qid] = dict(order=o, picks=list(dict.fromkeys(picks)), raw=raw[qid])
            study_helpers.atomic(lf, res)
            study_helpers.timing(tag, run, secs, study_helpers.peak_gb(), questions=len(qs))
        L = json.load(open(lf))
        sub = {}
        for q in qs:
            qid = q["qid"]; c = G["cands"][qid]; s = G["scores"][KEY[bk]][qid]
            boost = {x: 1000 - r for r, x in enumerate(L[qid]["order"][i] for i in L[qid]["picks"])}
            sub[qid] = study_helpers.build(F, q, rn, c, [boost.get(x, v) for x, v in zip(c, s)])
        study_helpers.put(tag, split, run, sub, qs)
    study_helpers.log("done", split, run, [t for t, _ in todo])
