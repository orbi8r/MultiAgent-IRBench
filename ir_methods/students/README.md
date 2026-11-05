# students

Small cross encoders that read candidates much faster than the Qwen3 rerankers. They are trained on public rows only. GPU for training, CPU is possible for scoring a short list.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `train_student.py BASE OUTTAG ROWS_GLOB` | trains a MiniLM class student with hard labels (`MAXLEN` env sets the input length, 256 by default), then scores the benchmark pools and times serving | `data/pub_*/train_synth.jsonl` | `models/<OUTTAG>`, tags `ce_*` and `rstul*` |
| `train_cross_encoder.py BASE OUTDIR EPOCHS MAXLEN FILE...` | the same for a bge base cross encoder. A row carrying `teacher` scores trains against that distribution with a KL loss | rows | the model folder |
| `teacher_scores.py BASE LORA OUT SHARD NSHARDS D FILE...` | teacher scores of a public reader on public rows, one shard at a time | rows | jsonl of scores |
| `make_teacher_rows.py OUT.jsonl` | joins the 4B score plus half the 0.6B score into one teacher per row | `teach/t4b_*.jsonl`, `teach/t06_*.jsonl` | OUT.jsonl, the rows for `ds_bgeb` |
| `score_student.py RUN MODELDIR TAG` | scores fuse0 top 100 with a trained student | `data/<run>/f1.json` | `data/<run>/<TAG>sub1.json` |

Recipes: `recipes/test5/10_train_public.sh`, `recipes/test5/40_pool_and_students.sh`.
