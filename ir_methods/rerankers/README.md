# rerankers

The reader stage: take the candidates of the first stage and give each a score. Zero shot Qwen3 Rerankers, LoRA adapters trained on public traces, cascades, listwise readers. Run with `$IRBENCH_WORK` as the working directory, GPU needed unless noted.

## Training

| Script | What it does |
|---|---|
| `train_reranker.py BASE OUTDIR EPOCHS ROWS.jsonl...` | LoRA fine tune of a Qwen3 Reranker with a listwise softmax over 1 positive and G-1 hard negatives. Rows are `{query, pos, negs}`. Optional resumable checkpoints with `CKPT` and `CKEVERY` |
| `train_ftpub.py NAME DIV SEED` | trains the 0.6B ftpub recipe on every DIV th public row, used for the training size and seed ablations |

## Scoring

| Script | What it does | Writes |
|---|---|---|
| `gpu_score.py RUN TAG EMBEDDER RERANKERS [K]` | builds the z2 candidates (BM25 ctx, plain, dense, RRF) and scores them with each reranker. Model names may end in `+lora=<adapter folder>` | `data/<run>/gpu_<TAG>.json` |
| `score_z2.py GTAG "tag\|model\|mode"...` | scores the frozen z2 candidates and writes dev then test submissions, mode is `build` (list builder), `raw` (reader order) or `first` | `gpu_<GTAG>.json`, subs, rows, timing |
| `yes_no_scorer.py` | the shared yes/no scorer with a score cache, modes fp16, nf4 and CPU offload | imported |
| `score_pool.py RUN ADAPTER SHARD NSHARD` | scores a candidate pool with an ftpub style adapter, in shards | `pubsub1.sh<SHARD>.json` |
| `merge_shards.py PREFIX` | merges the shard files | `data/<run>/<PREFIX>.json` |
| `score_pool_4b.py NAME` | scores fuse0 top 200 and its expansion lists with a 4B adapter | tags `f2_<NAME>_*`, cache |
| `fusion.py` | first stage fusion ablations read by ftpub | tags `f2_fl_*`, `f2_exp_fuse0_*` |

## Cascades and listwise readers

| Script | What it does |
|---|---|
| `cascade.py RUN BASE_TAG BASE_MODEL BIG_MODEL N OUT_TAG` | a bigger reranker rereads the top N of a saved ranking, tag `c4b` |
| `reader_8b.py dev\|test` | zero shot 8B reader and the 8B cascade, tags `f2_z8b`, `f2_c8b` |
| `listwise.py RUN BASE_TAG BASE_MODEL LLM N OUT_TAG` | a local instruct model picks the pieces that state the answer, tag `lw4` |
| `listwise_multi.py LLM fp16\|nf4 TAG\|BASEKEY...` | the same with several tags sharing one model load, tags `f2_lw4*`, `f2_lw8*` |

Recipes: `recipes/test5/10_train_public.sh` (training), `recipes/test5/20_readers.sh`, `recipes/full2/`.
