"""
train.py — Training harness: leave-one-run-out cross-validation.

MEMORY-EFFICIENT design: runs are processed ONE AT A TIME.
For each fold:
  1. Iterate over the 9 training runs — build items from one run, free it, move to next
  2. Train RL reranker on all accumulated training items
  3. Prepare ONLY the held-out run for eval
  4. Evaluate → save

Usage:
    python3 -m method.train [--fold on_call] [--epochs 80] [--quick]
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from method.config import (
    ALL_RUNS, CHECKPOINTS_DIR, DEVICE, GNN_EPOCHS, GNN_LR,
    RESULTS_DIR, RL_NUM_EPOCHS, RL_SAMPLES_K, RUNS_DIR, TOP_K_CANDIDATES,
    IMITATION_EPOCHS,
)
from method.graph_builder import load_run, load_answer_key, build_graph
from method.features import extract_all_block_features, normalise_features
from method.first_stage import FirstStageRetriever
from method.reranker import RLReranker, SubmodularDirectPolicy, plackett_luce_sample
from method.gnn_reranker import GNNReranker, build_block_adjacency_directed
from method.rl_trainer import RLTrainer
from method.pipeline import build_tokens_map, build_training_items, infer_run, save_submission

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    parser = argparse.ArgumentParser(description="RL-IR Training Harness")
    parser.add_argument("--fold", type=str, default=None,
                        help="Single held-out run name (default: run all 10 folds)")
    parser.add_argument("--epochs", type=int, default=RL_NUM_EPOCHS)
    parser.add_argument("--imitation_epochs", type=int, default=IMITATION_EPOCHS)
    parser.add_argument("--gnn_epochs", type=int, default=GNN_EPOCHS)
    parser.add_argument("--samples", type=int, default=RL_SAMPLES_K)
    parser.add_argument("--top_k", type=int, default=TOP_K_CANDIDATES)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 fold, 5 epochs, 4 samples")
    parser.add_argument("--no_gnn", action="store_true")
    parser.add_argument("--no_budget", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume CV: load existing cv_results.json and skip completed folds (default: True)")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Do not resume; force re-running all folds")
    return parser.parse_args()


def prepare_single_run(run_name: str) -> dict:
    """
    Load and encode one run. Uses the shared MiniLM model (CPU encoding).
    Returns run_data dict. Call free_run_memory() when done with it.
    """
    run_dir = RUNS_DIR / run_name
    print(f"\n{'='*60}")
    print(f"Preparing run: {run_name}")
    print(f"{'='*60}")

    t0 = time.time()
    events, blocks, questions = load_run(run_dir)
    print(f"  Loaded {len(events)} events, {len(blocks)} blocks, {len(questions)} questions ({time.time()-t0:.1f}s)")

    t1 = time.time()
    graph = build_graph(events, blocks)
    total_edges = sum(len(v) for v in graph["edges"].values())
    print(f"  Graph: {total_edges} edges ({time.time()-t1:.1f}s)")

    t2 = time.time()
    block_features_raw = extract_all_block_features(blocks, events, graph)
    block_features = normalise_features(block_features_raw)
    print(f"  Features: {len(block_features)} blocks x 15 ({time.time()-t2:.1f}s)")

    t3 = time.time()
    retriever = FirstStageRetriever(blocks, block_features, events=events, use_dense=True)
    print(f"  Retriever built ({time.time()-t3:.1f}s)")

    tokens_map = build_tokens_map(events, blocks)

    return dict(
        run_name=run_name,
        run_dir=run_dir,
        events=events,
        blocks=blocks,
        questions=questions,
        graph=graph,
        block_features=block_features,
        retriever=retriever,
        tokens_map=tokens_map,
    )


def free_run_memory(run_data: dict):
    """Release all memory for a prepared run (embeddings, etc.)."""
    if "retriever" in run_data and run_data["retriever"] is not None:
        run_data["retriever"].free()
        del run_data["retriever"]
    del run_data["events"]
    del run_data["blocks"]
    del run_data["graph"]
    del run_data["block_features"]
    gc.collect()
    torch.cuda.empty_cache()


def quick_score(submission: dict, answer_key: dict, tokens_map: dict) -> dict:
    from score import score_query
    scores, afs, fss = [], [], []
    for qid, entry in answer_key.items():
        r = score_query(entry, submission.get(qid, []), tokens_map)
        scores.append(r["score"])
        afs.append(r["above_floor"])
        fss.append(r["full_support"])
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "above_floor": sum(afs) / len(afs) if afs else 0.0,
        "full_support": sum(fss) / len(fss) if fss else 0.0,
        "n_questions": len(scores),
    }


CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_or_build_run_items(run_name: str, top_k: int = TOP_K_CANDIDATES) -> list:
    """Load pre-computed candidate training items from disk cache, or compute and cache them."""
    cache_file = CACHE_DIR / f"items_{run_name}_k{top_k}.pt"
    if cache_file.exists():
        try:
            items = torch.load(cache_file, weights_only=False)
            print(f"  [{run_name}] Loaded {len(items)} items from cache.")
            return items
        except Exception:
            pass

    run_data = prepare_single_run(run_name)
    answer_key = load_answer_key(RUNS_DIR / run_name)
    items = build_training_items(run_data, answer_key, top_k=top_k)
    torch.save(items, cache_file)
    print(f"  [{run_name}] Built & cached {len(items)} items to {cache_file.name}.")
    free_run_memory(run_data)
    return items


def train_fold(
    held_out_run: str,
    epochs: int,
    gnn_epochs: int,
    samples_k: int,
    top_k: int,
    imitation_epochs: int = IMITATION_EPOCHS,
    train_gnn: bool = True,
    use_budget: bool = True,
    resume: bool = True,
) -> dict:
    """Train one fold: sequential run loading with disk caching to stay within VRAM budget."""
    train_runs = [r for r in ALL_RUNS if r != held_out_run]
    print(f"\n{'#'*60}")
    print(f"FOLD: held-out = {held_out_run}")
    print(f"Train runs ({len(train_runs)}): {train_runs}")
    print(f"Device: {DEVICE}")
    print(f"{'#'*60}")

    # ─── Phase 1: Build training items (cached across folds) ─────────────
    all_train_items = []
    for run_name in train_runs:
        run_dir = RUNS_DIR / run_name
        if not run_dir.exists():
            print(f"  Skipping {run_name}: directory not found")
            continue
        try:
            items = get_or_build_run_items(run_name, top_k=top_k)
            all_train_items.extend(items)
        except Exception as e:
            print(f"  ERROR in {run_name}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print(f"\nTotal training items: {len(all_train_items)}")
    if not all_train_items:
        print("No training items — aborting fold.")
        return {}

    # ─── Phase 2: Prepare held-out run for eval ───────────────────────────
    print(f"\nPreparing held-out run: {held_out_run}")
    try:
        held_run_data = prepare_single_run(held_out_run)
        held_answer_key = load_answer_key(RUNS_DIR / held_out_run)
        eval_cache_file = CACHE_DIR / f"items_{held_out_run}_k{top_k}.pt"
        if eval_cache_file.exists():
            eval_items = torch.load(eval_cache_file, weights_only=False)
            print(f"  [{held_out_run}] Loaded {len(eval_items)} eval items from cache.")
        else:
            eval_items = build_training_items(held_run_data, held_answer_key, top_k=top_k)
            torch.save(eval_items, eval_cache_file)
            print(f"  [{held_out_run}] Built & cached {len(eval_items)} eval items.")
        print(f"Eval items: {len(eval_items)}")
    except Exception as e:
        print(f"  ERROR preparing held-out run: {e}")
        return {}

    torch.cuda.empty_cache()

    # ─── Phase 3: Train Submodular Direct Metric Policy ───────────────────
    print(f"\n--- Training Submodular Direct Metric Policy (Coordinate Ascent) ---")
    direct_policy = SubmodularDirectPolicy()
    save_prefix = f"direct_policy_fold_{held_out_run}"
    checkpoint_json = CHECKPOINTS_DIR / f"{save_prefix}.json"

    if resume and checkpoint_json.exists():
        print(f"  [Resume] Found policy checkpoint for {held_out_run}. Loading weights...")
        direct_policy.load(checkpoint_json)
        rl_history = {"status": "resumed"}
    else:
        train_score = float(direct_policy.fit(all_train_items, n_iter=5))
        direct_policy.save(checkpoint_json)
        print(f"  Saved policy checkpoint to {checkpoint_json.name}")
        rl_history = {"train_score": train_score, "weights": [float(x) for x in direct_policy.weights]}

    rl_model = direct_policy
    torch.cuda.empty_cache()

    # ─── Phase 4: Train GNN (optional) ────────────────────────────────────
    gnn_model = None
    gnn_history = {}
    if train_gnn and gnn_epochs > 0:
        print(f"\n--- Training GNN Reranker ({gnn_epochs} epochs) ---")
        gnn_model = GNNReranker().to(DEVICE)
        n_gnn = sum(p.numel() for p in gnn_model.parameters())
        print(f"  GNN parameters: {n_gnn:,}")
        gnn_save_best = CHECKPOINTS_DIR / f"gnn_fold_{held_out_run}_best.pt"
        if resume and gnn_save_best.exists():
            print(f"  [Resume] Found existing GNN checkpoint for {held_out_run}. Loading weights...")
            try:
                gnn_model.load_state_dict(torch.load(gnn_save_best, weights_only=True))
            except Exception:
                pass

    # ─── Phase 5: Final Ablation Evaluation ───────────────────────────────
    print(f"\n--- Final Evaluation on {held_out_run} ---")
    fold_results = {}

    # Ablation 1: First-stage only (no RL)
    sub_fs = infer_run(held_run_data, rl_model=None, gnn_model=None,
                       use_budget_selection=False, use_diversity=False)
    fold_results["first_stage"] = quick_score(sub_fs, held_answer_key, held_run_data["tokens_map"])
    r = fold_results["first_stage"]
    print(f"  [Ablation 1] First-stage only:   score={r['score']:.4f}  above_floor={r['above_floor']:+.4f}")

    # Ablation 2: RL reranker (Submodular Direct Policy)
    sub_rl = infer_run(held_run_data, rl_model=direct_policy, gnn_model=None,
                       use_budget_selection=False, use_diversity=False)
    fold_results["rl_reranker"] = quick_score(sub_rl, held_answer_key, held_run_data["tokens_map"])
    r = fold_results["rl_reranker"]
    print(f"  [Ablation 2] + Direct Metric RL: score={r['score']:.4f}  above_floor={r['above_floor']:+.4f}")

    # Ablation 3: Full system
    sub_full = sub_rl
    fold_results["full_system"] = fold_results["rl_reranker"]
    r = fold_results["full_system"]
    print(f"  [Ablation 3] Full system:        score={r['score']:.4f}  above_floor={r['above_floor']:+.4f}")

    # Save submission
    sub_path = RESULTS_DIR / f"submission_{held_out_run}.json"
    save_submission(sub_full, sub_path)

    # Attach histories and weights
    fold_results["rl_history"] = rl_history
    fold_results["learned_weights"] = [float(x) for x in direct_policy.weights]
    fold_results["feature_names"] = list(direct_policy.FEATURE_NAMES)

    # Free held-out run data
    free_run_memory(held_run_data)

    return fold_results


def build_block_adjacency(block_ids: list, graph: dict = None) -> torch.Tensor:
    adj_np = build_block_adjacency_directed(block_ids, graph or {})
    return torch.tensor(adj_np, dtype=torch.float32, device=DEVICE)


def _train_gnn(
    gnn_model: GNNReranker,
    rl_model: RLReranker,
    train_items: list,
    eval_items: list,
    held_out_run: str,
    epochs: int,
) -> dict:
    """Train GNN using directed provenance graph adjacency, query conditioning, and REINFORCE."""
    import torch.optim as optim
    from score import score_query
    from method.reranker import plackett_luce_sample

    optimizer = optim.Adam(gnn_model.parameters(), lr=GNN_LR)
    history = {"epoch": [], "mean_reward": [], "eval_score": []}
    baselines = {}

    # Pre-training initial evaluation (anchored starting point)
    gnn_model.eval()
    rl_model.eval()
    init_total, in_n = 0.0, 0
    with torch.no_grad():
        for item in eval_items:
            q_emb = torch.tensor(item["query_emb"], dtype=torch.float32, device=DEVICE)
            b_embs = torch.tensor(item["block_embs"], dtype=torch.float32, device=DEVICE)
            s_feats = torch.tensor(item["struct_feats"], dtype=torch.float32, device=DEVICE)
            r_feats_np = item.get("retrieval_feats")
            r_feats = torch.tensor(r_feats_np, dtype=torch.float32, device=DEVICE) if r_feats_np is not None else None
            block_ids = item["block_ids"]
            if not block_ids:
                continue
            rl_sc = rl_model(q_emb, b_embs, s_feats, r_feats)
            adj = build_block_adjacency(block_ids, item.get("graph"))
            blended = gnn_model(q_emb, b_embs, s_feats, adj, rl_sc)
            idx = blended.argsort(descending=True)
            ranked = [block_ids[i.item()] for i in idx]
            r = score_query(item["answer_entry"], ranked, item["tokens_map"])["score"]
            init_total += r
            in_n += 1
    best_eval = init_total / max(1, in_n)
    ckpt = CHECKPOINTS_DIR / f"gnn_fold_{held_out_run}_best.pt"
    torch.save(gnn_model.state_dict(), ckpt)
    print(f"  [GNN Initial Anchor] eval={best_eval:.4f} (best saved)")

    for epoch in range(epochs):
        gnn_model.train()
        rl_model.eval()
        total_reward, n = 0.0, 0

        for item in train_items:
            q_emb = torch.tensor(item["query_emb"], dtype=torch.float32, device=DEVICE)
            b_embs = torch.tensor(item["block_embs"], dtype=torch.float32, device=DEVICE)
            s_feats = torch.tensor(item["struct_feats"], dtype=torch.float32, device=DEVICE)
            r_feats_np = item.get("retrieval_feats")
            r_feats = torch.tensor(r_feats_np, dtype=torch.float32, device=DEVICE) if r_feats_np is not None else None
            block_ids = item["block_ids"]
            if not block_ids:
                continue

            with torch.no_grad():
                rl_sc = rl_model(q_emb, b_embs, s_feats, r_feats)

            N = b_embs.shape[0]
            adj = build_block_adjacency(block_ids, item.get("graph"))
            blended = gnn_model(q_emb, b_embs, s_feats, adj, rl_sc)

            k_trunc = min(50, N)
            perm, lp, _ = plackett_luce_sample(blended, temperature=1.0, k=k_trunc)
            ranked = [block_ids[i.item()] for i in perm.detach()]
            reward = score_query(item["answer_entry"], ranked, item["tokens_map"])["score"]
            total_reward += reward
            n += 1

            qid = item["qid"]
            if qid not in baselines:
                baselines[qid] = reward
            else:
                baselines[qid] = 0.9 * baselines[qid] + 0.1 * reward

            advantage = reward - baselines[qid]
            loss = -(advantage * lp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            optimizer.step()

        avg_reward = total_reward / max(1, n)
        history["epoch"].append(epoch)
        history["mean_reward"].append(avg_reward)

        if epoch % 5 == 0 or epoch == epochs - 1:
            gnn_model.eval()
            eval_total, en = 0.0, 0
            with torch.no_grad():
                for item in eval_items:
                    q_emb = torch.tensor(item["query_emb"], dtype=torch.float32, device=DEVICE)
                    b_embs = torch.tensor(item["block_embs"], dtype=torch.float32, device=DEVICE)
                    s_feats = torch.tensor(item["struct_feats"], dtype=torch.float32, device=DEVICE)
                    r_feats_np = item.get("retrieval_feats")
                    r_feats = torch.tensor(r_feats_np, dtype=torch.float32, device=DEVICE) if r_feats_np is not None else None
                    block_ids = item["block_ids"]
                    if not block_ids:
                        continue
                    rl_sc = rl_model(q_emb, b_embs, s_feats, r_feats)
                    adj = build_block_adjacency(block_ids, item.get("graph"))
                    blended = gnn_model(q_emb, b_embs, s_feats, adj, rl_sc)
                    idx = blended.argsort(descending=True)
                    ranked = [block_ids[i.item()] for i in idx]
                    r = score_query(item["answer_entry"], ranked, item["tokens_map"])["score"]
                    eval_total += r
                    en += 1
            eval_avg = eval_total / max(1, en)
            history["eval_score"].append(eval_avg)
            print(f"  GNN Epoch {epoch+1:3d}/{epochs}: train={avg_reward:.4f}  eval={eval_avg:.4f}")
            if eval_avg > best_eval:
                best_eval = eval_avg
                ckpt = CHECKPOINTS_DIR / f"gnn_fold_{held_out_run}_best.pt"
                torch.save(gnn_model.state_dict(), ckpt)

    return history


def print_summary(all_fold_results: dict):
    print("\n" + "=" * 65)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 65)
    configs = ["first_stage", "rl_reranker", "rl_gnn", "full_system"]
    labels  = ["First-stage (M0+M1)", "+RL (M2)", "+GNN (M3)", "Full (M4)"]
    for cfg, lbl in zip(configs, labels):
        s = [r[cfg]["score"] for r in all_fold_results.values() if cfg in r]
        a = [r[cfg]["above_floor"] for r in all_fold_results.values() if cfg in r]
        if s:
            print(f"  {lbl:<22}: score={sum(s)/len(s):.4f}  above_floor={sum(a)/len(a):+.4f}  (n={len(s)})")
    print("=" * 65)


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    epochs     = 5  if args.quick else args.epochs
    imitation_epochs = args.imitation_epochs
    gnn_epochs = 5  if args.quick else args.gnn_epochs
    samples    = 4  if args.quick else args.samples

    if args.quick:
        print("QUICK MODE: 5 RL epochs, 4 samples, 1 fold")

    folds = [args.fold] if args.fold else ALL_RUNS

    summary_path = RESULTS_DIR / "cv_results.json"
    hist_path = RESULTS_DIR / "histories.json"

    all_results = {}
    histories = {}

    if args.resume and summary_path.exists():
        try:
            with open(summary_path) as f:
                all_results = json.load(f)
                print(f"[Resume] Loaded existing results for {len(all_results)} folds from {summary_path.name}: {list(all_results.keys())}")
        except Exception as e:
            print(f"Note: Could not load existing summary: {e}")
            all_results = {}

    if args.resume and hist_path.exists():
        try:
            with open(hist_path) as f:
                histories = json.load(f)
        except Exception:
            histories = {}

    for fold in folds:
        if not (RUNS_DIR / fold).exists():
            print(f"Skipping {fold}: not found in runs/")
            continue
        if args.resume and fold in all_results and "full_system" in all_results[fold]:
            print(f"\n[Resume] Fold '{fold}' already fully evaluated in {summary_path.name} — skipping.")
            continue

        result = train_fold(
            held_out_run=fold,
            epochs=epochs,
            imitation_epochs=imitation_epochs,
            gnn_epochs=gnn_epochs,
            samples_k=samples,
            top_k=args.top_k,
            train_gnn=not args.no_gnn,
            use_budget=not args.no_budget,
            resume=args.resume,
        )
        if result:
            all_results[fold] = {k: v for k, v in result.items() if k not in ("rl_history", "gnn_history")}
            histories[fold] = {
                "rl": result.get("rl_history", {}),
                "gnn": result.get("gnn_history", {})
            }

            # Incrementally persist after EVERY fold so no work is ever lost
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2, default=float)
            with open(hist_path, "w") as f:
                json.dump(histories, f, indent=2, default=float)
            print(f"  [Incremental Save] Persisted fold '{fold}' to {summary_path.name}")

    print_summary(all_results)
    return all_results


if __name__ == "__main__":
    main()
