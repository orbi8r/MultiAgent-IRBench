"""Converts public agent traces (nebius/SWE-agent-trajectories on Hugging Face) into pseudo team records in the benchmark's format,
so a reranker can be trained on records that are not benchmark data.
Each pseudo run bundles T trajectories, one per member (agent dev_<i>, with its own conversation and shift), in sequence.
Ai turns become a model event (reasoning) and a tool_call event (the command), and the following user turn becomes a tool_result.
Blocks hold up to 8 lines and about 120 tokens.
    python convert_public_traces.py OUTDIR NRUNS T"""
import hashlib, json, os, re, sys
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

out, nruns, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OFF = int(os.environ.get("OFFSET", 0))  # skip this many picked trajectories (a second disjoint batch)
K0 = int(os.environ.get("K0", 0))  # first pseudo run number
api = HfApi()
files = sorted(f for f in api.list_repo_files("nebius/SWE-agent-trajectories", repo_type="dataset") if f.endswith(".parquet"))
rows = []
for f in files:
    tb = pq.read_table(hf_hub_download("nebius/SWE-agent-trajectories", f, repo_type="dataset"), columns=["instance_id", "trajectory"])
    rows += tb.to_pylist()
    if len(rows) >= (nruns * T + OFF) * 30:
        break
# deterministic pick, one trajectory per instance, spread by hash
seen, pick = set(), []
for r in sorted(rows, key=lambda r: hashlib.md5(r["instance_id"].encode()).hexdigest()):
    if r["instance_id"] in seen or not (20 <= len(r["trajectory"]) <= 120):
        continue
    seen.add(r["instance_id"]); pick.append(r)
ntok = lambda t: max(1, len(t) // 4)


def blocks_of(eid, text):
    out, cur = [], []
    for line in text.split("\n"):
        if not line.strip():
            continue
        cur.append(line)
        if len(cur) >= 8 or ntok("\n".join(cur)) >= 120:
            out.append("\n".join(cur)); cur = []
    if cur:
        out.append("\n".join(cur))
    return [dict(id=f"{eid}/{hashlib.md5((eid + str(i) + b).encode()).hexdigest()[:16]}", event=eid, text=b, tokens=ntok(b))
            for i, b in enumerate(out)]


for k in range(nruns):
    d = f"{out}/pub_{K0 + k:02d}"
    os.makedirs(d, exist_ok=True)
    ev, bl, n, ts = [], [], 0, 1.8e9
    for i, r in enumerate(pick[OFF + k * T:OFF + (k + 1) * T]):
        agent, conv = f"dev_{i}", hashlib.md5(r["instance_id"].encode()).hexdigest()
        for s, t in enumerate(r["trajectory"]):
            text = (t.get("text") or t.get("system_prompt") or "").strip()
            if not text:
                continue
            parts = []
            if t["role"] == "system" or (t["role"] == "user" and s <= 1):
                parts = [("prompt", text)]
            elif t["role"] == "ai":
                cmd = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.S)
                thought = re.sub(r"```.*?```", "", text, flags=re.S).strip()
                if thought:
                    parts.append(("model", thought))
                if cmd:
                    parts.append(("tool_call", "run_command(" + json.dumps({"CommandLine": cmd[-1].strip()}) + ")"))
            else:
                parts = [("tool_result", text[:20000])]
            for kind, txt in parts:
                n += 1; ts += 7
                eid = f"n{n:06d}"
                ev.append(dict(id=eid, agent=agent, kind=kind, ts=ts, shift=i + 1, conv=conv, text=txt, tokens=ntok(txt)))
                if kind != "prompt":
                    bl += blocks_of(eid, txt)
    with open(f"{d}/events.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in ev)
    with open(f"{d}/blocks.jsonl", "w") as f:
        f.writelines(json.dumps(b) + "\n" for b in bl)
    with open(f"{d}/questions.jsonl", "w") as f:
        pass
    print(d, len(ev), "events", len(bl), "blocks", sum(e["tokens"] for e in ev), "tokens", flush=True)
