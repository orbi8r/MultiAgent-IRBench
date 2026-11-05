"""P4: the original Model A (neural RLReranker, hidden 128, anchor 4.0, select_mmr gamma 0.03; Plackett Luce REINFORCE) under our protocol.
Train on the other dev runs per fold (LORO), all dev runs for test; last epoch, NO eval selection, NO held out data seen in training.
Declared settings (repo defaults of the original pipeline): imitation epochs 0 (config IMITATION_EPOCHS), RL epochs 15 (the original reranker_history.json), samples 16, seed 0.
    python fit_model_a.py OUT"""
import hashlib, json, pickle, shutil, sys, time, os
from pathlib import Path
import numpy as np
import torch

from rl_common import (DEV, TEST, DEDUPS, FEATS, load_key, write_sub, freeze, frozen, write_row, add_timing, peak_gb, dev_table, fmt_dev)
import score as S
from method.reranker import RLReranker
from patched_rl_trainer import RLTrainer

OUT = Path(sys.argv[1])
EPOCHS = int(os.environ.get("MODELA_EPOCHS", "15"))
SAMPLES = 16
SEED = 0
SEC = "S1 RL style reruns (one metric, dev only fits)"
TMP = Path(os.environ.get("IRBENCH_SCRATCH", ".")) / "pl_modelA_tmp"
dev_cuda = "cuda" if torch.cuda.is_available() else "cpu"
_runs = {}


def side(r, dd):
    if (r, dd) not in _runs:
        _runs[(r, dd)] = (pickle.load(open(FEATS / f"items_{r}_{dd}.pkl", "rb")), S.load_run(f"runs/{r}"))
    return _runs[(r, dd)]


def train_items(r, dd):
    items, run = side(r, dd)
    key = load_key(r)
    return [dict(it, tokens_map=run["tokens"], answer_entry=key[q], credits=S.credits(key[q], run), run=run) for q, it in items.items() if q in key]


def rank(model, r, dd):
    items, run = side(r, dd)
    sub = {q: model.select_mmr(block_ids=it["block_ids"], query_emb=it["query_emb"], block_embs=it["block_embs"],
                               struct_feats=it["struct_feats"], retrieval_feats=it["retrieval_feats"], tokens_map=run["tokens"])
           for q, it in items.items()}
    for q in json.load(open(FEATS / f"fs_{r}_{dd}.json")):
        sub.setdefault(q, [])
    return sub


def make():
    return RLReranker(hidden_dim=128, anchor_weight=4.0)


def load(path):
    m = make().to(dev_cuda)
    m.load_state_dict(torch.load(path, map_location=dev_cuda))
    return m


for dd in DEDUPS:
    tag = f"pl_modelA_{dd}"
    cache = {r: train_items(r, dd) for r in DEV}
    ck = OUT / "modelA_ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    for fold in DEV + ["all3"]:
        path = ck / f"{dd}_{fold}.pt"
        if path.exists():  # resume marker
            continue
        tr = [r for r in DEV if r != fold]
        data = [it for r in tr for it in cache[r]]
        torch.manual_seed(SEED); np.random.seed(SEED)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        model = make()
        trainer = RLTrainer(model, checkpoint_dir=TMP / f"{dd}_{fold}")
        trainer.train(data, eval_data=None, num_epochs=EPOCHS, imitation_epochs=0, save_prefix="m", samples_k=SAMPLES)
        torch.save(model.state_dict(), path)
        shutil.rmtree(TMP / f"{dd}_{fold}", ignore_errors=True)
        add_timing(OUT, tag, fold, time.time() - t0, peak_gb())
        print(dd, fold, "trained on", tr, round(time.time() - t0), "s", flush=True)
    for r in DEV:
        write_sub(OUT, "dev", tag, r, rank(load(ck / f"{dd}_{r}.pt"), r, dd))
    freeze(OUT, tag, dict(declared=True, dedup=dd, model="RLReranker hidden 128 anchor 4.0, select_mmr gamma 0.03", imitation_epochs=0, rl_epochs=EPOCHS,
                          samples=SAMPLES, seed=SEED, eval_data=None, checkpoint="last epoch, no eval selection", fit_runs=DEV,
                          checkpoint_file=f"modelA_ckpt/{dd}_all3.pt", sha256=hashlib.sha256(open(ck / f"{dd}_all3.pt", "rb").read()).hexdigest(),
                          evidence="dev runs only; no setting chosen by any score"))
    fz = frozen(OUT, tag)
    m = load(OUT / fz["checkpoint_file"])
    for r in TEST:
        write_sub(OUT, "test", tag, r, rank(m, r, dd))
    res = dev_table(OUT, tag, S)
    write_row(OUT, tag, section=SEC, method=f"P4 RL style Model A (neural RL reranker, Plackett Luce REINFORCE), dedup {dd}", doc_kind="LORO",
              frozen_setting=f"hidden 128, anchor 4.0, imitation 0, RL {EPOCHS} epochs, {SAMPLES} samples, seed {SEED}, last epoch, no eval selection, fit on 3 dev runs, dedup {dd}",
              clean_reason="trained on dev runs only with no eval selection; first stage candidates carry the original old era constants: design exposed",
              dev_note=fmt_dev(res) + " (leave one run out)",
              notes="the original shipped reranker_best.pt was chosen by eval score on a held out fold and its training data is unstated, so it is not reused; one seed only (the original seed noise on other variants was about 0.055)")
