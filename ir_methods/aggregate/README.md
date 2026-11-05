# aggregate

Scoring and the table builders. This is the only stage that reads answer keys.

| Script | What it does | Writes |
|---|---|---|
| `score_test_subs.py [tag ...]` | scores the frozen test submissions `$IRBENCH_OUT/subs/<tag>/<run>.json` of the five test runs with the repo scorer. A tag is complete only with all 5 runs and no missing question | `$IRBENCH_OUT/scores/<tag>.json`, one printed line per tag |
| `score_submissions.py [--subs SUBDIR] [--out-name NAME]` | scores every submission under `$IRBENCH_WORK/out/*/SUBDIR/{dev,test}/<tag>/<run>.json` (SUBDIR default `subs`), including per question values. `TASKGLOB` narrows the task folders, `DEVR` and `TESTR` the runs. A tag whose score file is newer than its submissions is skipped. The baselines call it with `--subs b3subs --out-name b3scores` | `$IRBENCH_OUT/NAME/{dev,test}/<tag>.json` |
| `aggregate_full2.py` | builds the main study table: scores, rows and timing of every tag, bootstrap 95 percent CI over questions | `results_full2.csv` and `.md`, per run and question type files, summary |
| `aggregate_full3.py` | builds the structure aware study table with paired gains over the flat controls (bootstrap, 2000 draws, seed 0) | `results_full3.csv` and `.md`, summary |
| `aggregate_baselines.py` | builds the baseline table from `$IRBENCH_WORK/out/*_b3_*/{b3rows,b3timing,b3scores}`, exit code 0 when all 22 tags are complete on all 8 runs | `$IRBENCH_OUT/results_full3b.csv` |
| `csv_to_md.py <csv>` | renders a results csv as a short markdown table (tag, method, per run scores) | markdown on stdout |
| `clean_flags.csv` | tag pattern to flag and reason (clean, not_run and so on), read by `aggregate_full2.py` and `aggregate_full3.py` |

The csv files that these scripts produced for the document are kept in `results/`. A tag that appears in several files keeps the row of the last one, which is the rule of the document build.
