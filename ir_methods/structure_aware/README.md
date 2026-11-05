# structure_aware

Structure aware variants (tags `f3_*` and `t3_*`). The document calls them inspired by STAIR, a generative retriever. They are plain reranking variants and do not implement STAIR, which is why the folder is not named after it. The idea is to give the reader or the ranker the structure of the run: which event, file, agent shift and section a piece belongs to. Two routes: sectional reranking of cached reader scores (CPU), and a reader trained on prefixed text (GPU). Nothing here reads a key except the dev scoring of `dev_gate.py` and `sectional_lib.py`.

Structure and data

| Script | What it does |
|---|---|
| `run_structure.py` | nodes of a run (event, file, agent plus shift, section) and the prefix strings P1 to P3 that put the structure in front of the reader. Keyless, works on real and public pseudo runs |
| `write_prefixes.py RUN [ROOT]` | writes `data/<run>/f3_pfx_<style>.json`, the prefixed text of every piece |
| `structure_stats.py OUTFILE RUN...` | structure statistics per run and prefix token lengths |
| `prefix_training_rows.py STAGE REFROWS OUTROOT "P1 P2"` | public training rows with prefixed text. The flat rebuild must equal the original rows |

Sectional reranking (CPU, on cached scores)

| Script | What it does |
|---|---|
| `sectional_lib.py` | helpers: method C adds a node score to the block score, method D learns a LightGBM lambdarank over node aggregates |
| `sectional_rerank.py 06\|4b` | runs both methods over the cached 0.6B scores or the 4B cache. Leave one run out on dev, one frozen config scores the test runs once |
| `cache_pool_scores_4b.py NAME` | reruns the 4B reader pool to keep the per pair score cache `sectional_rerank.py 4b` needs |

Trained prefix reader (GPU)

| Script | What it does |
|---|---|
| `train_prefix_reader.py NAME STYLE SEED [BASE]` | trains a reader on prefixed public rows (4964 rows, 1 epoch) with the same recipe as the plain fine tune |
| `score_prefix_reader.py --tag T --model M --style S --splits ...` | scores the frozen z2 candidates with prefixed document text |
| `train_score_06b.sh STYLE SEED SPLITS [freeze]` | train then score for the 0.6B reader |
| `score_pool_4b_prefix.py STYLE` | the same at 4B over fuse0 top 200 and the expansion lists |

Gate and controls

| Script | What it does |
|---|---|
| `dev_gate.py` | dev only gate: go only if the dev mean beats the flat control by the declared gain and is not worse on 2 of 3 dev runs |
| `pick_prefix_style.py STYLES SEED` | picks the better style on dev, freezes it, scores test only when the gate says go |
| `control_tags.py [--tags]` | the flat control tags of the main study for each `f3_` tag, used for paired gains |

Recipes: `recipes/full3/` in numeric order. The table is `results/results_full3.csv`.
