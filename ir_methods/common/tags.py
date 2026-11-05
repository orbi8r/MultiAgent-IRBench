"""Registry of the full2 study tags: (section, method, one line note). The row writer reads it; frozen settings come from each step's own script."""
import study_helpers

S1, S2, S3, S4, S5, S6, S7, S8, S9 = ("9.1 full2 first stage and readers", "9.2 full2 public training size and seeds", "9.3 full2 bigger readers and students",
                                       "9.4 full2 listwise", "9.5 full2 fast pipeline configs", "9.6 full2 timing only", "9.7 full2 reader body length",
                                       "9.8 full2 first stage fusion", "9.9 full2 4B readers on fuse0 pools")
Z2 = "z2 candidates (200 of RRF of ctx BM25, plain BM25, Qwen3 embedding 0.6B dense)"
META = {
    "f2_z2_first": (S1, "z2 first stage alone, top 200 in fused order", "no reader, key free, the same list every reader row below reranks"),
    "f2_z2_zs06": (S1, "zero shot Qwen3 Reranker 0.6B over z2 K200, list builder", "dev and per qtype reproduction of zs06"),
    "f2_z2_zs06_raw": (S1, "zero shot Qwen3 Reranker 0.6B over z2 K200, reader order without the list builder", "isolates the list builder"),
    "f2_z2_ftpub": (S1, "public trace ftpub over z2 K200, list builder", "dev and per qtype reproduction of ftpub"),
    "f2_z2_ftpub_raw": (S1, "public trace ftpub over z2 K200, reader order without the list builder", "isolates the list builder"),
    "f2_pub25": (S2, "ftpub recipe on every 4th public row (1241 of 4964)", "same recipe as ftpub, one epoch, fewer public rows, K200 over z2"),
    "f2_pub50": (S2, "ftpub recipe on every 2nd public row (2482 of 4964)", "same recipe as ftpub, one epoch, fewer public rows, K200 over z2"),
    "f2_ftpub_s1": (S2, "ftpub recipe retrained with seed 1", "LoRA init, dropout and row order change, data and settings as ftpub, K200 over z2"),
    "f2_ftpub_s2": (S2, "ftpub recipe retrained with seed 2", "LoRA init, dropout and row order change, data and settings as ftpub, K200 over z2"),
    "f2_ftpub_s3": (S2, "ftpub recipe retrained with seed 3", "LoRA init, dropout and row order change, data and settings as ftpub, K200 over z2"),
    "f2_maxb320": (S7, "ftpub over z2 K200 with the body capped at 320 tokens", "inference cap only, the adapter is ftpub"),
    "f2_maxb640": (S7, "ftpub over z2 K200 with the body capped at 640 tokens", "inference cap only, the adapter is ftpub"),
    "f2_z8b": (S3, "zero shot Qwen3 Reranker 8B over z2 K200", "8B in the memory mode frozen in the file (4 bit nf4 or fp16 with CPU offload), 4 bit can move scores slightly"),
    "f2_c8b": (S3, "cascade: 8B rereads the zs06 top 50 of z2", "built from the f2_z8b scores, same pairs and prompt as cascade.py, no rerun"),
    "f2_c4b": (S3, "cascade: 4B rereads the zs06 top 50 of z2", "cascade.py, dev and per qtype counterpart of c4b"),
    "f2_ds_bgeb": (S3, "student ds_bgeb over z2 K200", "cross encoder trained on public rows, dev and per qtype counterpart of ds_bgeb"),
    "f2_lw4": (S4, "listwise Qwen3 4B Instruct reads the zs06 top 10 of z2", "picks first, then the zs06 order"),
    "f2_lw4pub": (S4, "listwise Qwen3 4B Instruct reads the ftpub top 10 of z2", "picks first, then the ftpub order"),
    "f2_lw8": (S4, "listwise Qwen3 8B (nf4) reads the zs06 top 10 of z2", "thinking off, picks first, then the zs06 order"),
    "f2_lw8pub": (S4, "listwise Qwen3 8B (nf4) reads the ftpub top 10 of z2", "thinking off, picks first, then the ftpub order"),
}
for c, why in (("K50", "ftpub on fuse0 top 50"), ("X50", "early exit, ftpub on top 20 and up to 50 when unsure"), ("X200", "early exit, ftpub on top 20 and up to 200 when unsure"),
               ("S50_20", "ds_bgeb student on fuse0 top 50, ftpub reads the student top 20"),
               ("S200_30", "ds_bgeb student on fuse0 top 200, ftpub reads the student top 30")):
    META[f"f2_fast_{c}"] = (S5, f"fast pipeline config {c}: {why}", "timed_run.py, one process per run, fuse0 first stage, list builder")
for t, why in (("t2_ftpub_k200", "ftpub on fuse0 top 200, timed_run K200 total without model load"),
               ("t2_c4b", "lexical + dense + zs06 over K200 + 4B over the top 50, scoring only"),
               ("t2_ds_bgeb", "lexical + dense + ds_bgeb over K200, scoring only")):
    META[t] = (S6, why, "timing only, no sub")
FL = ("first1", "first3", "first5", "no_bm_ctx", "no_bm_plain", "no_bm_raw", "no_bm_nov", "no_rm3", "no_qdecomp", "no_qtype")
for v in FL:
    META[f"f2_fl_{v}_k100"] = (S8, f"first stage fusion variant {v}, top 100 read by ftpub", "RRF k 60 over the pass 1 lists of the variant, key free")
    META[f"f2_fl_{v}_k100_m25"] = (S8, f"first stage fusion variant {v}, top 100 read by ftpub plus the top 25 of each expansion list", "seeds are the reader top 5, expansion as ftpub_exp rows")
for m in (50, 100):
    META[f"f2_exp_fuse0_k200_m{m}"] = (S8, f"ftpub on fuse0 top 200 plus the top {m} of each expansion list", "extends the m5 m10 m25 grid")
for n, why in (("ftpub4b", "4B ftpub4b"), ("ftpub4b2x", "4B ftpub4b2x (two batches of public rows)")):
    META[f"f2_{n}_fuse0_k200"] = (S9, f"{why} over fuse0 top 200", "first stage fuse0 instead of z2")
    META[f"f2_{n}_exp_fuse0_k200_m25"] = (S9, f"{why} over fuse0 top 200 plus the top 25 of each expansion list", "seeds are the 4B top 5, expansion as ftpub_exp rows")


def row(tag, kind, frozen, dev_note=""):
    sec, method, note = META[tag]
    study_helpers.row(tag, sec, method, kind, frozen, dev_note, note)
