# RL-Based Retrieval Method

A **reinforcement learning** retrieval system for MultiAgent-IRBench that directly optimises the benchmark's graded coverage metric using REINFORCE with Plackett-Luce sampling.

## Architecture

```
events.jsonl + blocks.jsonl
         │
    [Module 0] Provenance Graph Construction
         │  5 edge types: flow, result, file, copy, ref
         ▼
    [Module 1] Structure-Aware First-Stage Retrieval
         │  BM25 + Dense (MiniLM) + Structural bonus → Weighted RRF → top-200
         ▼
    [Module 2] RL-Trained Reranker (core contribution)
         │  REINFORCE + Plackett-Luce sampling → optimises score_query() directly
         ▼
    [Module 3] Graph-Aware GNN Reranker
         │  2-layer GAT propagates relevance along provenance edges
         ▼
    [Module 4] Budget-Aware Selection
         │  Value-density ordering + same-event diversity penalty
         ▼
    submission.json
```

## Quick Start

```bash
# Install dependencies
pip3 install torch sentence-transformers numpy scikit-learn matplotlib --break-system-packages

# Quick test (1 fold, 5 epochs, 4 samples)
python3 -m method.train --quick --fold on_call

# Full training (all 10 folds, 80 epochs)
python3 -m method.train --epochs 80

# Generate plots and ablation table
python3 -m method.evaluate --strata
```

## File Structure

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `graph_builder.py` | Module 0: Provenance graph (flow/result/file/copy/ref edges) |
| `features.py` | 15-d structural features per block |
| `first_stage.py` | Module 1: BM25 + Dense + structural RRF |
| `reranker.py` | Module 2: 215K-param scoring head + Plackett-Luce |
| `rl_trainer.py` | REINFORCE training loop with temperature annealing |
| `gnn_reranker.py` | Module 3: 2-layer GAT graph reranker |
| `budget_selector.py` | Module 4: Value-density + diversity selection |
| `pipeline.py` | End-to-end inference pipeline |
| `train.py` | Leave-one-run-out CV training harness |
| `evaluate.py` | Ablations, strata analysis, learning curve plots |

## Training Details

- **Cross-validation**: Leave-one-run-out (10 folds × ~50 questions each)
- **RL algorithm**: REINFORCE with Plackett-Luce ranking policy
- **Reward**: `score_query()` — the exact benchmark metric
- **Temperature**: Annealed from τ=1.0 → τ=0.1 over 80 epochs
- **Trainable params**: ~215K (RL head) + ~100K (GNN)
- **MiniLM**: Frozen (saves VRAM on RTX 3050 4GB)
- **Estimated time**: ~1-2 hours total

## Expected Results

| Configuration | Expected Score | Above Floor |
|--------------|---------------|-------------|
| Fusion baseline | 0.30–0.50 | +0.15–0.40 |
| M0+M1 (Struct+Fuse) | 0.35–0.55 | +0.20–0.45 |
| +M2 (RL) | 0.45–0.65 | +0.30–0.55 |
| +M3 (GNN) | 0.48–0.68 | +0.33–0.58 |
| Full System | 0.50–0.70 | +0.35–0.60 |
