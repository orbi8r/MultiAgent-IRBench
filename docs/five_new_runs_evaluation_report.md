# Evaluation Report: Testing the Master Reranker on the 5 New Runs (Zero Training)

## 1. Executive Summary

We evaluated the trained **Master Submodular Direct Policy** (calibrated solely on the first 3 runs) strictly out-of-domain on the **5 newly published rebuilt runs** (`due_diligence`, `executive_assistants`, `help_desk`, `on_call`, and `textbook_team` — 250 questions total).

**Zero training or adaptation was performed on these 5 runs.**

### Key Outcomes:
- **100% Win Rate**: The model outperforms the first-stage baseline on **every single one of the 5 new runs** (5 / 5).
- **Macro Main BAR Score**: Lifted from **0.4013 $\to$ 0.4485** (**+0.0472** average lift).
- **Macro Equal Credit Score**: Lifted from **0.4740 $\to$ 0.5440** (**+0.0700** average lift).
- **Early Budget Retrieval**: Over **50.4%** of queries across all 5 new domains have every answer atom identified within $\le 2,000$ tokens, and **63.2%** within $\le 32,000$ tokens.

---

## 2. Head-to-Head Comparison: Baseline vs. Model

Evaluated using the official Budgeted Answer Recall (BAR) scorer (`python score.py runs/<run> submission_<run>.json`):

| Domain | Questions | First-Stage Baseline Main | **Master Model Main** | **Main Score Lift** | First-Stage Baseline Equal | **Master Model Equal** | **Equal Credit Lift** | Answered $\le$ 2k Tokens | Answered $\le$ 32k Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`due_diligence`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/due_diligence) | 50 | 0.3987 | **0.4709** | **+0.0722** | 0.5200 | **0.6350** | **+0.1150** | 54.0% | 58.0% |
| [`executive_assistants`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/executive_assistants) | 50 | 0.4659 | **0.5178** | **+0.0519** | 0.5400 | **0.6000** | **+0.0600** | 60.0% | 72.0% |
| [`help_desk`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/help_desk) | 50 | 0.3896 | **0.4027** | **+0.0130** | 0.4350 | **0.4400** | **+0.0050** | 44.0% | 56.0% |
| [`on_call`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/on_call) | 50 | 0.3172 | **0.3570** | **+0.0398** | 0.3500 | **0.4450** | **+0.0950** | 40.0% | 58.0% |
| [`textbook_team`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/textbook_team) | 50 | 0.4350 | **0.4941** | **+0.0591** | 0.5250 | **0.6000** | **+0.0750** | 54.0% | 72.0% |
| **5 NEW RUNS MEAN** | **250** | **0.4013** | **0.4485** | **+0.0472** | **0.4740** | **0.5440** | **+0.0700** | **50.4%** | **63.2%** |

---

## 3. Full 8-Run Macro Benchmark Picture

Combining the 3 original runs with the 5 new rebuilt runs (all 8 published runs in MultiAgent-IRBench, 396 questions in total):

| Domain | Total Questions | Main BAR Score | Equal Credit Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| [`library_maintainers`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/library_maintainers) | 47 | 0.5863 | 0.5851 | Original Run |
| [`literature_review`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/literature_review) | 50 | 0.4411 | 0.5250 | Original Run |
| [`online_shop`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/online_shop) | 49 | 0.5060 | 0.5816 | Original Run |
| [`due_diligence`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/due_diligence) | 50 | 0.4709 | 0.6350 | New Rebuilt Run |
| [`executive_assistants`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/executive_assistants) | 50 | 0.5178 | 0.6000 | New Rebuilt Run |
| [`help_desk`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/help_desk) | 50 | 0.4027 | 0.4400 | New Rebuilt Run |
| [`on_call`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/on_call) | 50 | 0.3570 | 0.4450 | New Rebuilt Run |
| [`textbook_team`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/runs/textbook_team) | 50 | 0.4941 | 0.6000 | New Rebuilt Run |
| **8-RUN BENCHMARK MEAN** | **396** | **0.4720** | **0.5515** | **All Published Runs** |

---

## 4. Observations on Generalization

1. **Robust Transfer Across Domains**:
   Without seeing a single question or answer key from the 5 new runs during training, the model achieves substantial gains:
   - `due_diligence`: **+0.0722** Main / **+0.1150** Equal Credit.
   - `textbook_team`: **+0.0591** Main / **+0.0750** Equal Credit.
   - `executive_assistants`: **+0.0519** Main / **+0.0600** Equal Credit (crossing 0.51).
2. **Stability of the Learned Policy**:
   The relaxed anchor ($\text{prior}=1.0$), combined with dense-lexical synergy (`field_bm25` + `dense*bm25`), location file-path boost, and micro-budget reordering, consistently lifts nuggets out of deep ranks regardless of domain-specific jargon.
3. **Artifacts & Reproducibility**:
   - Submissions: [`method/results/submission_{due_diligence,executive_assistants,help_desk,on_call,textbook_team}.json`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/results)
   - Official Reports: [`method/results/report_{due_diligence,executive_assistants,help_desk,on_call,textbook_team}.json`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/results)
   - Evaluation Script: [`test_new_runs.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/test_new_runs.py)
   - Structured Results: [`method/results/five_new_runs_evaluation.json`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/project/MultiAgent-IRBench/method/results/five_new_runs_evaluation.json)
