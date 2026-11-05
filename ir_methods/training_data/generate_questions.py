"""Writes synthetic training questions from the record alone (no keys, no benchmark questions). A local instruct model reads one
piece with its context and writes a question that a team member might later ask, whose answer that piece states.
    python generate_questions.py RUN N MODEL
Writes data/<run>/synth.jsonl = {target, question, answer, kind}.
"""
import json, re, sys, hashlib, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

run, N, mname = sys.argv[1], int(sys.argv[2]), sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else "fact"
P = [json.loads(l) for l in open(f"data/{run}/pieces.jsonl")]


def useful(p):
    t = p["text"]
    return 15 <= p["tokens"] <= 300 and p["kind"] in ("tool_call", "tool_result", "model") and len(re.findall(r"[A-Za-z]{3,}", t)) >= 4


pool = [p for p in P if useful(p)]
# deterministic spread: order by a hash of the id, then fill per kind shares
import os
SALT, PRIOR, TAG = os.environ.get("SALT", ""), os.environ.get("PRIOR", ""), os.environ.get("STAG", "")
if PRIOR == "copies":
    # pieces whose text shows up again in a later step were looked up again: the kind of fact teams ask about later
    norm = lambda t: " ".join(t.split())
    blob = "\x00".join(norm(p["text"]) for p in P)
    def later(p):
        t = norm(p["text"])
        return len(t.split()) >= 5 and blob.count(t) > 1
    pool = [p for p in pool if later(p)]
pool.sort(key=lambda p: hashlib.md5((SALT + p["id"]).encode()).hexdigest())
share = {"tool_result": 0.45, "tool_call": 0.4, "model": 0.15} if MODE == "fact" else {"tool_call": 1.0}
pick = []
for k, s in share.items():
    pick += [p for p in pool if p["kind"] == k][: int(N * s)]

EXAMPLES = """Examples of the style (made up, about other teams):
- What retry limit did the deploy script end up using after the timeout fix?
- In the weekly sales summary, which product line had the largest drop?
- Where did the analyst save the cleaned survey data?
- Which agent first proposed switching the cache to Redis, and in which shift?
- Why was the vendor's second invoice rejected?
- According to the onboarding notes, what must you run before opening a pull request?"""

SYS = "You write test questions for a search engine over the saved work record of a team of AI agents."


def prompt(p):
    if MODE == "who":
        return (f"{EXAMPLES}\n\nHere is one step of the record: an action a team member took, with its context.\n---\n"
                f"{p['ctx'][:1200]}\n---\nWrite one question asking which team member took this action and in which shift. "
                "Describe the action in other words (what was changed, committed, written, moved or run, and on what) without "
                "naming who did it, so that it has one answer in a very large record. One sentence.\n"
                f'Reply with JSON only: {{"question": "...", "answer": "{p["agent"]}, shift {p["shift"]}"}}')
    return (f"{EXAMPLES}\n\nHere is one piece of the record and its context.\n---\n{p['ctx'][:1200]}\n---\n"
            "Write one question a member of the team might later ask whose answer is a specific fact stated in this piece "
            "(a number, name, path, command, decision, reason, status or date). The piece alone must answer it. Paraphrase: "
            "use the words a teammate would use, not long phrases copied from the piece. Give enough context (which file, "
            "report, issue, paper, thread or step) that the question has one answer in a very large record. One sentence.\n"
            'Reply with JSON only: {"question": "...", "answer": "short answer"}')


tok = AutoTokenizer.from_pretrained(mname, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.float16).cuda().eval()
texts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": prompt(p)}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in pick]
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
out = [None] * len(texts)
t0 = time.time()
bs = 8
for s in range(0, len(order), bs):
    ch = order[s:s + bs]
    enc = tok([texts[i] for i in ch], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=90, do_sample=False, pad_token_id=tok.pad_token_id)
    for i, row in zip(ch, g[:, enc["input_ids"].shape[1]:]):
        out[i] = tok.decode(row, skip_special_tokens=True)
    if s % (bs * 20) == 0:
        print(s, len(order), f"{time.time() - t0:.0f}s", flush=True)
kept = 0
with open(f"data/{run}/synth{'' if MODE == 'fact' else '_' + MODE}{TAG}.jsonl", "w") as f:
    for p, o in zip(pick, out):
        m = re.search(r"\{.*\}", o or "", re.S)
        try:
            j = json.loads(m.group(0))
            q, a = j["question"].strip(), str(j["answer"]).strip()
        except Exception:
            continue
        aw = set(re.findall(r"[a-z0-9]+", a.lower()))
        tw = set(re.findall(r"[a-z0-9]+", p["text"].lower()))
        if MODE == "who":
            tw = set(re.findall(r"[a-z0-9]+", p["ctx"].lower()))
        if not aw or len(aw & tw) / len(aw) < 0.6 or len(q) < 25:
            continue
        f.write(json.dumps(dict(target=p["id"], question=q, answer=a, kind=p["kind"])) + "\n")
        kept += 1
print("DONE", run, "kept", kept, "of", len(pick), f"{time.time() - t0:.0f}s")
