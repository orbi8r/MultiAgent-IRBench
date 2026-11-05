# fast_pipeline

The whole method as one process, timed end to end per run. Nothing is fitted on the runs and no key is read.

| Script | What it does | Writes |
|---|---|---|
| `timed_run.py RUN ADAPTER CONFIGS OUT` | tokenize, novelty, BM25 indexes, fuse0 first stage for all questions, the public reader (merged LoRA, fp16), then the list builder. Configs such as `S100_30`, `X100`, `K50` set reader depth. Reports total = shared stages + the config's own | OUT.json with `times` and `lists` |
| `timed_run_subs.py [TAG]` | turns the `timed_run.py` outputs of the five test runs into submissions and a timing file | `$IRBENCH_OUT/subs/fast_<config>/<run>.json`, `fast/times_<TAG>.json` |
| `timed_run_configs.py CFG[,CFG]` | runs the same configs over all runs and keeps only subs and timing | tags `f2_fast_*` |
| `cascade_timing.py c4b\|t2\|fast` | the c4b cascade subs and the timing only tags `t2_*` | frozen files, rows |
| `run_method.py RUN_DIR OUT.json [--k 100] [--model M] [--lora PATH]` | the standalone method of tag `t1`: contextual pieces, BM25 and plain BM25 by RRF, Qwen3 Reranker on every visible GPU, list builder | the submission |

Recipes: `recipes/test5/50_fast_pipeline.sh`, `recipes/full2/37_fast_cfgs.sh`, `recipes/full2/38_timing_c4b_ds.sh`.
