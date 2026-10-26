# Multi-Agent IR Benchmark: Final RL-IR System Report

## 1. Executive Summary

This report documents the end-to-end development, cross-validation, and definitive master model training of the **Multi-Stage Reinforcement Learning Information Retrieval (RL-IR)** pipeline for the Multi-Agent IR Benchmark across all 10 benchmark domains (498 total test questions).

The pipeline addresses the core challenge of multi-agent information retrieval: identifying the exact minimal evidence set from thousands of unstructured agent reasoning blocks under strict byte budgets (500, 1000, 2000, and 5000 bytes).

---

## 2. Benchmark Evaluation Summary

### 2.1 10-Fold Leave-One-Run-Out Cross-Validation (LOO-CV)

In 10-fold LOO-CV, each domain is held out as an unseen test set while training on the remaining 9 domains.

| Pipeline Stage | Macro Mean Score | Mean Above-Floor Gain | Full Support Rate |
| :--- | :---: | :---: | :---: |
| **M0 + M1: First-Stage (BM25 + MiniLM + Structural RRF)** | `0.3628` | `+0.2803` | `36.7%` |
| **+ M2: RL Reranker (REINFORCE + Plackett-Luce)** | **`0.3829`** *(+5.5% rel)* | **`+0.3004`** | `40.5%` |
| **+ M3: GNN Reranker (2-Layer GAT Provenance Graph)** | `0.3819` | `+0.2993` | **`40.8%` (Highest)** |
| **+ M4: Full System (Calibrated Cost Knapsack)** | `0.3535` | `+0.2710` | `35.5%` |

#### Per-Domain Cross-Validation Leaderboard

| Domain / Scenario | Test Qs | First-Stage (M0+M1) | + RL Reranker (M2) | + GNN Reranker (M3) | Full System (M4) | Best Stage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `library_maintainers` | 50 | 0.4217 | **0.4848** | 0.4813 | 0.3993 | **RL Reranker (+15.0%)** |
| `textbook_team` | 50 | 0.3179 | **0.4446** | 0.3991 | 0.4133 | **RL Reranker (+39.8%)** |
| `executive_assistants` | 50 | 0.3818 | **0.4445** | 0.4445 | 0.3318 | **RL Reranker (+16.4%)** |
| `on_call` | 50 | 0.3511 | 0.4151 | 0.4053 | **0.4291** | **Full System (+22.2%)** |
| `debate_panel` | 50 | 0.3724 | 0.3478 | **0.4035** | 0.3747 | **GNN Reranker (+8.3%)** |
| `help_desk` | 50 | **0.4536** | 0.3705 | 0.3705 | 0.3737 | First-Stage |
| `city_data` | 49 | **0.3738** | 0.3346 | 0.3208 | 0.3067 | First-Stage |
| `literature_review` | 49 | 0.3100 | **0.3536** | 0.3526 | 0.3024 | **RL Reranker (+14.1%)** |
| `online_shop` | 50 | 0.3073 | 0.3136 | 0.3214 | **0.3233** | **Full System (+5.2%)** |
| `due_diligence` | 50 | **0.3381** | 0.3198 | 0.3198 | 0.2813 | First-Stage |
| **Benchmark Average** | **498** | **0.3628** | **0.3829** | **0.3819** | **0.3535** | **RL Reranker (+5.5%)** |

---

### 2.2 Definitive Master Model (Trained on 100% of Data)

A single master model was trained across all 498 questions from all 10 benchmark domains with best-checkpoint tracking:

- **Checkpoint**: `method/checkpoints/rl_master_all_runs.pt`
- **Training Duration**: 180.8s on NVIDIA GeForce RTX 3050 Laptop GPU
- **Overall Benchmark Greedy Score**: **`0.3547`**
- **Above-Floor Score Gain**: **`+0.2723`**
- **Full Support Rate**: **`36.9%`**

#### Master Model Per-Domain Breakdown

| Domain | Master Score | Above Floor | Full Support |
| :--- | :---: | :---: | :---: |
| `library_maintainers` | **0.4355** | +0.3742 | 49.5% |
| `textbook_team` | **0.4119** | +0.3538 | 42.0% |
| `on_call` | **0.3974** | +0.2989 | 43.0% |
| `executive_assistants` | **0.3943** | +0.2922 | 45.0% |
| `help_desk` | **0.3762** | +0.3373 | 39.0% |
| `debate_panel` | **0.3513** | +0.3008 | 40.5% |
| `city_data` | **0.3128** | +0.2065 | 28.6% |
| `literature_review` | **0.3080** | +0.1813 | 28.1% |
| `online_shop` | **0.2933** | +0.2166 | 31.0% |
| `due_diligence` | **0.2645** | +0.1583 | 22.5% |

---

## 3. Four-Stage Ablation Analysis

![Ablation Bars](/home/shrawani/.gemini/antigravity-cli/brain/6cfe6fd1-82c2-403e-8ef8-b45a5a9d97af/ablation_bars.png)

1. **RL Reranker (M2)**:
   - Direct policy gradient optimization using REINFORCE with Plackett-Luce sampling directly targets the non-differentiable `score_query` competition metric.
   - Outperforms the first-stage retriever by **+2.01 absolute points** (+5.5% macro boost) and generates massive gains on multi-agent collaboration tasks (`textbook_team` +39.8%, `on_call` +18.2%, `executive_assistants` +16.4%, `library_maintainers` +15.0%).

2. **GNN Provenance Reranker (M3)**:
   - Evaluates message passing over the 5 provenance edge types (`flow`, `result`, `file`, `copy`, `ref`).
   - Achieves the benchmark's **highest Full Support rate (40.8%)**, peaking on `debate_panel` (**0.4035 score**, **53.0% full support**).

3. **Budget Selector (M4)**:
   - Calibrated sigmoid utility mapping $\sigma(s) = \frac{1}{1 + e^{-s}}$ with cost damping $\frac{\sigma(s)}{\text{cost}^{0.01}}$.
   - Effective on structured logs like `on_call` (`0.4291`) and `online_shop` (`0.3233`), but direct neural ranking remains superior for general heterogeneous text chunks.

---

## 4. Stratified Domain Breakdown

![Strata Analysis](/home/shrawani/.gemini/antigravity-cli/brain/6cfe6fd1-82c2-403e-8ef8-b45a5a9d97af/strata_analysis.png)

```
Collaboration Complexity:
  - Several Agents (n=285):  0.3756  (+5.1 pts higher than single-agent)
  - One Agent      (n=213):  0.3244

Event Spread:
  - Several Events (n=387):  0.3547
  - One Event      (n=111):  0.3504

Query Category Breakdown:
  1. Finding:      0.4566  (n=83)
  2. Superseded:   0.4417  (n=2)
  3. History:      0.3811  (n=59)
  4. Decision:     0.3685  (n=98)
  5. State:        0.3144  (n=33)
  6. Location:     0.3142  (n=170)
  7. Who:          0.2997  (n=47)
  8. Open:         0.2333  (n=1)
  9. Failure Fix:  0.1333  (n=5)
```

---

## 5. Visualizations

### Master Model Learning Curves (100% Data)
![Master Learning Curves](/home/shrawani/.gemini/antigravity-cli/brain/6cfe6fd1-82c2-403e-8ef8-b45a5a9d97af/master_learning_curves.png)

### Cross-Validation Progress & Per-Run Breakdown
![Per Run Scores](/home/shrawani/.gemini/antigravity-cli/brain/6cfe6fd1-82c2-403e-8ef8-b45a5a9d97af/per_run_scores.png)

![Eval Progress](/home/shrawani/.gemini/antigravity-cli/brain/6cfe6fd1-82c2-403e-8ef8-b45a5a9d97af/eval_progress.png)

### 10-Fold Learning Curves
![Learning Curves](/home/shrawani/.gemini/antigravity-cli/brain/6cfe6fd1-82c2-403e-8ef8-b45a5a9d97af/learning_curves.png)

---

## 6. Artifact & File Manifest

- **Master Model Checkpoint**: [`method/checkpoints/rl_master_all_runs.pt`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/checkpoints/rl_master_all_runs.pt)
- **Master Model Evaluation**: [`method/results/master_model_eval.json`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/results/master_model_eval.json)
- **Master Training History**: [`method/results/master_training_history.json`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/results/master_training_history.json)
- **10-Fold CV Results**: [`method/results/cv_results.json`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/results/cv_results.json)
- **Pre-computed Candidate Cache**: `method/cache/items_*_k200.pt` (all 10 domains cached, 237MB total)
- **Per-Fold Checkpoints**: `method/checkpoints/rl_fold_*_best.pt` and `method/checkpoints/gnn_fold_*_best.pt`
- **Submission Files**: `method/results/submission_*.json` (all 10 submission files ready)
