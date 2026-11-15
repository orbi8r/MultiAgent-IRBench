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

# Full training (one fold per run in runs/, 80 epochs)
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

- **Cross-validation**: Leave one run out (one fold per run, about 50 questions each)
- **RL algorithm**: REINFORCE with Plackett-Luce ranking policy
- **Reward**: `score_query()`, the exact benchmark metric
- **Temperature**: Annealed from τ=1.0 → τ=0.1 over 80 epochs
- **Trainable params**: ~215K (RL head) + ~100K (GNN)
- **MiniLM**: Frozen (fits a 4 GB GPU)

## Results

This pipeline was first built for the earlier 10 run release and its old scorer. Its methods were run again under the current `score.py` on the 8 public runs as part of the IR study. Those numbers are in `docs/MultiAgentBench_IR_study.docx` and `ir_methods/results/`, and the scripts that ran them are in `ir_methods/rl_style/`.
