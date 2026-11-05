# baselines

The off the shelf baselines of the study (tags `b3_*`, doc section on lexical and dense baselines) plus the plain contextual BM25 list. No training and no tuning. Every setting is declared in `b3_lib.py`.

| Script | What it does | Writes |
|---|---|---|
| `b3_lib.py` | shared helpers: piece loading, tokenizers, BM25 variants, the list builder variants, frozen files, rows, timing | imported |
| `b3_lex.py` | lexical and trivial baselines over every piece of every run, CPU only: BM25 with whitespace, word, ident and stem tokens, each with and without the list builder, plus random, recent, oldest, short and long | `$IRBENCH_OUT/b3subs/{dev,test}/<tag>/<run>.json`, `b3rows`, `b3timing` |
| `b3_dense.py` | four off the shelf embedders (each visits the GPU alone) and the RRF of BM25 stem with bge base, with and without the list builder. `B3_MODELS` narrows the models | the same folders |
| `bm25_ctx_alone.py [run ...]` | tag `bm25_ctx_alone`: the contextual BM25 list on its own, read from `data/<run>/lex.json` of the first stage | `$IRBENCH_OUT/subs/bm25_ctx_alone/<run>.json` |

Scoring is `aggregate/score_submissions.py` (the same scorer wrapper every study uses), called by `recipes/full3b/44_b3_score.sh` as `TASKGLOB='*_b3_*' score_submissions.py --subs b3subs --out-name b3scores`, which writes `$IRBENCH_OUT/b3scores/{dev,test}/<tag>.json`.

Recipes: `recipes/full3b/42_b3_lex.sh`, `43_b3_dense.sh`, `44_b3_score.sh`, `99_aggregate.sh`, and the end of `recipes/test5/60_make_subs_and_score.sh` for `bm25_ctx_alone`. `99_aggregate.sh` runs `aggregate/aggregate_baselines.py`, which joins the rows, timing and scores of all `*_b3_*` task folders into `results_full3b.csv` in its own `$IRBENCH_OUT`.

`recipes/00_prep_lexical.sh` first writes the per run piece files that `b3_lex.py` reads. The lexical half needs no GPU and runs in about a minute on one public run, which is the smoke test of the main README.
