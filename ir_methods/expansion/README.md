# expansion

Document and query expansion for the BM25 reader. All models are local and off the shelf, nothing is fitted and no key is read by the methods.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `doc2query_generate.py RUN t5\|qwen` | doc2query expansion of every piece: `t5` is castorini/doc2query-t5-base-msmarco with 5 sampled queries, `qwen` is Qwen3 1.7B with one greedy question | `data/<run>/pieces.jsonl` | `data/<run>/d2q_t5.jsonl` or `d2q_qwen.jsonl` |
| `doc2query_reader.py t5 qwen` | BM25 reader over expanded pieces plus the list builder, in the concat form and in the field form (`bm_ctx + L bm_exp`, L 0.5 or 1.0) | the `d2q_*.jsonl` files | `out/subs/d2q_<variant>/<run>.json`, tags `d2q_*` |
| `conversation_prf.py` | pseudo relevance across the conversation: a piece gets its BM25 z score plus B times the best z score of neighbouring pieces in the same conversation, 5 events either side. Writes the whole grid as submissions | `data/<run>/` lexical files | `$SUBDIR/convprf_<cell>/<run>.json`, tags `convprf_*` |

Recipe: `recipes/test5/30_cpu_families.sh`. Generating the doc2query files is the slow step, a GPU makes it a few minutes per run.
