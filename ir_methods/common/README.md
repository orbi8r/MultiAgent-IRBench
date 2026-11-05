# common

Code that every stage imports. Scripts add this folder to their path, so run them from `$IRBENCH_WORK` with `python3 $IR/<stage>/<script>.py`.

| File | What it is |
|---|---|
| `run_loader.py` | run loader and BM25. Reads the public files of a run (`events.jsonl`, `blocks.jsonl`, `questions.jsonl`) from `$IRBENCH_RUNS`. The answer key is opened only when `IRBENCH_KEYS=1`, which only the scoring and diagnostic scripts set. Puts the repo root on the path so `import score` finds the repo scorer |
| `study_helpers.py` | helpers of the main study steps: dev then test plan, the list builder call, frozen settings files, submissions, result rows, timing and the per tag score cache. Opens no answer key |
| `tags.py` | registry of the study tags as `(doc section, method, one line note)`. The row writer reads it, and `results/tag_index.csv` is its reader friendly form |

Environment variables are listed in the main README.
