# first_stage

Turns a run into contextual pieces and a short candidate list per question. No reader and no answer key is involved, except in the recall scripts that say so.
Every script runs with `$IRBENCH_WORK` as the working directory and reads or writes `data/<run>/`.

## Core

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `contextual_pieces.py` | builds the contextual piece text (who wrote it, step kind, command or file, event head) and the BM25 helpers | a run's blocks and events | imported, no output |
| `lexical_candidates.py RUN[,RUN]` | contextual pieces and the lexical lists, the entry point of every pipeline | `$IRBENCH_RUNS/<run>` | `data/<run>/pieces.jsonl`, `lex.json` (top 300 per question), `questions.jsonl` |
| `novelty.py` | novelty of a piece, the share of word 5 gram shingles not seen earlier | pieces | imported |
| `bm25_tools.py` | BM25 index, rank and RRF helpers and a plain against contextual comparison | pieces, keys for scoring | prints per run recall |

z2 is the first stage of the readers: BM25 over context, BM25 over plain text and the dense Qwen3 Embedding 0.6B list, fused by RRF, top 200. The dense part lives in `rerankers/gpu_score.py`. fuse0 is the CPU only first stage: the RRF of seven lexical and query side lists, top 100.

## Algorithmic first stage families

| Script | What it does | Writes |
|---|---|---|
| `cpu_families.py` | lexical, query side and conversation structure families, top 2000 each | `data/<run>/fam_cpu.json` |
| `arch1.py` | pass 1 lists, then two small LightGBM selectors, leave one run out | `fam_arch.json`, `arch1_time.json` |
| `arch2.py` | interleaved compound first stage with feedback rounds | `fam_arch2.json`, `arch2_time.json` |
| `arch2_rrf.py` | the key free part of arch2 only (rounds fused by RRF, no LightGBM) | `fam_arch2.json` {arch2_rrf, pass1_rrf}, timing |
| `write_fuse0.py` | writes fuse0 top 100 and its timing | `fuse0.json`, `fuse0_time.json` |

## Pools, recall and timing

| Script | What it does | Writes |
|---|---|---|
| `reader_pools.py 1\|2` | key free candidate pools for the public reader: round 1 is the union of base and fuse0 top 200, round 2 adds the top 25 of each expansion list seeded by the reader's top 5 | `pool1.json`, `pool2.json`, `f1.json` |
| `merge_reader_pools.py` | merges the two reader pool rounds and checks that every top 200 candidate is scored | `ftpub_pool.json` |
| `family_pool.py` | candidate pool of every family for the reader | `fam5_lists.json`, `fam5_pool.json` |
| `first_stage_recall.py [variant]` | recall of first stage variants and of the novelty pruned list, keys used only to measure | prints the `fs_*` rows |
| `family_recall.py OUT.json` | recall at 200 of every key free family | OUT.json, the `rec200_*` rows |
| `first_stage_timing.py` | timing tag `t2_z2_first`, lexical plus dense seconds per run | `timing/t2_z2_first.json` |

Recipes that call these: `recipes/00_prep_lexical.sh`, `recipes/full2/10_prep.sh`, `recipes/test5/30_cpu_families.sh`, `recipes/test5/40_pool_and_students.sh`.
