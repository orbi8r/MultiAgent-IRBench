"""
config.py — All paths, hyperparameters, and constants for the RL-IR retrieval system.
"""
import os
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
METHOD_DIR = PROJECT_ROOT / "method"
CHECKPOINTS_DIR = METHOD_DIR / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = METHOD_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_RUNS = [
    d.name for d in sorted(RUNS_DIR.iterdir())
    if d.is_dir() and (d / "events.jsonl").exists() and (d / "questions.jsonl").exists()
] if RUNS_DIR.exists() else [
    "library_maintainers",
    "literature_review",
    "online_shop",
]

# ─── Module 0: Graph construction ─────────────────────────────────────────
BOILERPLATE_THRESHOLD = 12      # hashes in >12 events = boilerplate
MIN_BLOCK_CHARS = 30            # shorter blocks skipped for copy detection
COMMON_LOCATOR_THRESHOLD = 15   # locators in >15 events = too generic
LISTING_THRESHOLD = 15          # events with >15 locators = directory listing

# ─── Module 1: First-stage retrieval ──────────────────────────────────────
TOP_K_CANDIDATES = 200          # top-K candidates passed to reranker
BM25_K1 = 1.2
BM25_B = 0.75
DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DENSE_BATCH_SIZE = 128
# Structural bonus weights (trained by grid search)
STRUCT_WEIGHTS = {
    "kind_is_model":       0.15,
    "has_file_write_edge": 0.10,
    "has_file_read_edge":  0.08,
    "degree_norm":         0.05,
    "position_frac":       0.07,
}
RRF_K = 60                      # RRF constant

# ─── Module 2: RL Reranker ─────────────────────────────────────────────────
RL_EMBED_DIM = 384              # MiniLM output dimension
RL_STRUCT_DIM = 15              # structural feature vector size
RL_STRUCT_PROJ_DIM = 64         # projected struct dimension
RL_HIDDEN_DIM = 64              # score head hidden layer (compact to prevent memorization)
RL_DROPOUT = 0.30               # strong regularization
RL_RETRIEVAL_FEAT_DIM = 10      # 10 first-stage scoring & cost signals per candidate

# RL training parameters
IMITATION_EPOCHS = 0            # disabled by default (imitation overfits to question-specific indices)
IMITATION_LR = 1e-3
RL_NUM_EPOCHS = 30
RL_SAMPLES_K = 16               # Plackett-Luce samples per question
RL_TEMP_START = 0.5             # temperature annealing start
RL_TEMP_END = 0.20              # temperature annealing end
RL_BASELINE_MOMENTUM = 0.99     # EMA momentum for reward baseline
RL_LR = 3e-4
RL_GRAD_CLIP = 1.0
RL_WEIGHT_DECAY = 1e-4

# Budget-aware selection parameters
MMR_LAMBDA = 0.0                # disable aggressive cosine penalty
MMR_GAMMA = 0.03                # optimal token cost penalty

# ─── Module 3: GNN ────────────────────────────────────────────────────────
GNN_HIDDEN_DIM = 128
GNN_NUM_HEADS = 4
GNN_NUM_LAYERS = 2
GNN_LR = 3e-4
GNN_EPOCHS = 40
GNN_SCORE_BLEND = 0.3           # weight of GNN score in final blend (1-α for RL score)

# ─── Module 4: Budget selector ────────────────────────────────────────────
BUDGETS = (1000, 2000, 4000, 8000)
MIN_COST = 64
BUDGET_COST_ALPHA = 0.01             # Calibrated sub-linear cost damping exponent
BUDGET_USE_DIVERSITY = False         # Same-event blocks frequently contain complementary nuggets
DIVERSITY_SAME_EVENT_PENALTY = [1.0, 1.0, 1.0]

# ─── Device ───────────────────────────────────────────────────────────────
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()
