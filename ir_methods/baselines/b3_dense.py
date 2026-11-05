"""b3 off the shelf dense baselines, the RRF of BM25 stem with bge base and their list builder variants (GPU).
All four embedders stay on the CPU between uses and visit the GPU one at a time so the peak per tag is its own.
Env B3_MODELS (comma separated tags) narrows the models for a smoke. Idempotent per run marker."""
import os, time
import numpy as np, torch
import b3_lib as B
import study_helpers

TASK = "dense"
M = {  # tag: (hf name, query prefix, passage prefix, sentence-transformers prompt_name for queries)
    "b3_dense_minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", "", None),
    "b3_dense_bgeb": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", "", None),
    "b3_dense_e5b": ("intfloat/e5-base-v2", "query: ", "passage: ", None),
    "b3_dense_qwen3e": ("Qwen/Qwen3-Embedding-0.6B", "", "", "query"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEL = [t for t in os.environ.get("B3_MODELS", ",".join(M)).split(",")]
LB = {"b3_dense_bgeb_lb": "b3_dense_bgeb", "b3_dense_qwen3e_lb": "b3_dense_qwen3e", "b3_rrf_stem_bgeb_lb": "b3_rrf_stem_bgeb"}
B.write_rows_and_freeze(B.DENSE_TAGS)

from sentence_transformers import SentenceTransformer
emb = {}
for tag in SEL:
    name = M[tag][0]
    emb[tag] = SentenceTransformer(name, device="cpu", model_kwargs={"torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32})
    emb[tag].max_seq_length = 512
    B.log("loaded", name)


def encode(tag, docs, queries):
    name, qp, dp, pn = M[tag]
    m = emb[tag].to(DEVICE)
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.time()
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    E = m.encode([dp + docs[i] for i in order], batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=DEVICE)
    E = E[torch.argsort(torch.tensor(order, device=E.device))]
    kw = {"prompt_name": pn} if pn else {}
    QE = m.encode([qp + q for q in queries], batch_size=32, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=DEVICE, **kw)
    sim = (QE.float() @ E.float().T).cpu().numpy()
    peak = 0.0
    if DEVICE == "cuda": torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated() / 2 ** 30
    secs = time.time() - t
    del E, QE; emb[tag] = m.to("cpu")
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return sim, secs, peak


def top(row, k):
    o = np.argsort(-row, kind="stable")[:k]
    return [int(i) for i in o]


for split, run in B.plan():
    if B.marker(TASK, split, run).exists():
        B.log("skip", split, run); continue
    t0 = time.time()
    qs, F, R, P = study_helpers.builder(run)
    ids = [p["id"] for p in P]
    docs, queries = [p["ctx"] for p in P], [B.qtext(q) for q in qs]
    res = {}  # tag -> (raw sub, cands per qid, secs, peak)
    sims = {}
    for tag in SEL:
        sim, secs, peak = encode(tag, docs, queries)
        sub, cands = {}, {}
        for j, q in enumerate(qs):
            o = top(sim[j], B.RAW_N)
            sub[q["qid"]] = [ids[i] for i in o]
            cands[q["qid"]] = ([ids[i] for i in o[:B.LB_N]], [float(sim[j][i]) for i in o[:B.LB_N]])
        res[tag] = (sub, cands, secs, peak)
        if tag == "b3_dense_bgeb":
            sims[tag] = sim
        B.log(run, tag, f"{secs:.1f}s peak {peak:.2f} GB")
    if "b3_dense_bgeb" in SEL:
        t = time.time()
        stem = B.bm25_stem_lists(P, qs, B.RRF_DEPTH)
        sub, cands = {}, {}
        for j, q in enumerate(qs):
            dl = top(sims["b3_dense_bgeb"][j], B.RRF_DEPTH)
            o, s = B.rrf([stem[q["qid"]][0], dl])
            sub[q["qid"]] = [ids[i] for i in o[:B.RAW_N]]
            cands[q["qid"]] = ([ids[i] for i in o[:B.LB_N]], s[:B.LB_N])
        # time of the fusion = stem BM25 + fusion on top of the bge base pass
        res["b3_rrf_stem_bgeb"] = (sub, cands, res["b3_dense_bgeb"][2] + time.time() - t, res["b3_dense_bgeb"][3])
    for tag, (sub, cands, secs, peak) in res.items():
        B.put(tag, split, run, sub, qs); B.timing(tag, run, secs, peak)
    for tag, src in LB.items():
        if src not in res:
            continue
        t = time.time()
        cands = res[src][1]
        sub = {q["qid"]: study_helpers.build(F, q, R, cands[q["qid"]][0], cands[q["qid"]][1]) for q in qs}
        B.put(tag, split, run, sub, qs); B.timing(tag, run, res[src][2] + time.time() - t, res[src][3])
    B.atomic(B.marker(TASK, split, run), dict(secs=round(time.time() - t0, 1), questions=len(qs), pieces=len(P)))
    B.log("done", split, run, len(P), "pieces", f"{time.time() - t0:.1f}s")
print("DENSE_ALL_DONE")
