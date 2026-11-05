# recipes

One shell script per step. Each is a thin wrapper that sets the run lists and calls the Python entry points with the settings that made the table rows. Read a recipe to see the exact recipe of a tag, and run it to reproduce the tag.

Shared files

| File | What it does |
|---|---|
| `lib.sh` | sourced first by every recipe. Sources `../env.sh`, sets `DEV` and `TEST` run lists, `IRBENCH_OUT=$IRBENCH_WORK/out/<study>_<recipe name>` (for example `out/full3b_42_b3_lex`, so two studies with a `99_aggregate.sh` never share a folder; recipes in the folder `recipes/` itself and the `test5/` study, which shares one `out/test5`, are the exceptions), model names and the working directory. `RUNSEL`, `DEVR` and `TESTR` narrow the runs (an empty `DEVR` or `TESTR` means none) |
| `run_all.sh FOLDER [LO HI]` | runs the recipes of one folder in numeric order and skips those with a `.done` marker `$IRBENCH_WORK/out/.done/<folder>_<recipe name>`. Delete the marker to run a recipe again. It does not read `NEEDS`, the numbers give the order |
| `00_prep_lexical.sh` | CPU. Writes `data/<run>/pieces.jsonl` and `lex.json` for every run. First step of the baselines and the smoke test |

Header lines inside a recipe: the first comment says what it makes and whether it needs a GPU. `# NEEDS=<recipe>` names recipes that must finish first (documentation for the reader, `run_all.sh` only follows the numbers). `# EST=<seconds>` is the measured or estimated time on one 11 GB GPU. `# LAST=1` marks the aggregate step that ends a study.

Folders

| Folder | Study | Content |
|---|---|---|
| `test5/` | the frozen test study of the five test runs | `lib.sh` plus six steps: train public readers and students, score readers, CPU families, pools and students, the fast pipeline, make submissions and score. Results in `results/results_test5.csv` |
| `full2/` | the main study with dev leave one run out and frozen test, ftpub family | numbered 10 to 99, prep to aggregate. Results in `results/results_full2.csv` |
| `full3/` | the structure aware variants | sectional reranking and prefix readers. Results in `results/results_full3.csv` |
| `full3b/` | the baselines on all eight runs | lexical, dense, score. Results in `results/results_full3b.csv` |
| `rl_style/` | the RL style methods re run | run from the repo root. Tags `pl_*`, in the `full2` results |

Order for a first run: `00_prep_lexical.sh`, then the folder you want. The GPU recipes need about 11 GB of GPU memory. They were not rerun for this repo on a small GPU, only the CPU recipes were (see the main README, Smoke test).
