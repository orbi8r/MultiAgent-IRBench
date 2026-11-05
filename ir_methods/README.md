# ir_methods

The code and recipes behind every result in the study document `docs/MultiAgentBench_IR_study.docx`.
Each result in that document carries a tag such as `ftpub`, `f2_lw4` or `b3_bm25_word`. For every tag this folder holds the code that made it, the recipe that ran it and the row with its numbers.
The lookup file is `results/tag_index.csv`: one line per tag, with the doc section, the entry point and the recipe. The document spells some script names the way they were first written (`subs5.py`, `famcpu.py`, `ftrain3.py`); `results/doc_script_names.csv` maps each to the file it has now.

Only the eight public runs are used (`runs/`): online_shop, library_maintainers and literature_review are the dev runs, on_call, due_diligence, executive_assistants, help_desk and textbook_team are the test runs. No method reads the key of a run it is scored on. Fits on the dev runs (rl_style, sectional method D, the probes) read only dev keys and say so in their header; the scorers and diagnostic scripts open keys to score.

## How the pipeline fits together

```
runs/<run>/{blocks,events,questions}.jsonl
   |
   v  first_stage/     contextual pieces, BM25 lists, dense list, fusion        -> top 100 or 200 candidates per question
   |
   v  rerankers/       zero shot or LoRA fine tuned reader, student, listwise   -> a score per candidate
   |
   v  list_builder/    drop repeats, earliest copy, location provenance solver  -> ordered piece ids per question
   |
   v  submission {question id: [piece ids]}  ->  score.py (repo root)  ->  score
```

Around that spine: `training_data/` makes the public training rows, `students/` and `fast_pipeline/` hold the small and fast readers, `expansion/` and `algorithmic/` hold the methods that need no neural reader, `baselines/` the plain baselines, `structure_aware/` the structure aware variants, `rl_style/` the RL style policies re run on our pool, `aggregate/` the tables.

## Directory map

| Folder | What is in it |
|---|---|
| `common/` | run loader and scorer wrapper (`run_loader.py`), shared plan and frozen file helpers (`study_helpers.py`), tag registry (`tags.py`) |
| `baselines/` | lexical, dense and trivial baselines on all eight runs (`b3_*`), BM25 over contextual pieces |
| `first_stage/` | contextual pieces, BM25 and dense candidate lists, fusion (z2, fuse0), recall and timing of the families |
| `rerankers/` | zero shot and LoRA readers: training (`train_reranker.py`), scoring (`gpu_score.py`), cascades, listwise readers |
| `students/` | small cross encoders trained on public rows, teacher scores, distillation |
| `training_data/` | public agent traces turned into pseudo team records, synthetic questions, training rows |
| `list_builder/` | the final list builder, the provenance solver, writers of frozen submissions |
| `expansion/` | document and query expansion (doc2query, pseudo relevance) |
| `algorithmic/` | readers with no neural model |
| `fast_pipeline/` | the whole method as one timed process |
| `structure_aware/` | structure prefixes, sectional rerank, gated variants |
| `rl_style/` | the RL style policies re run under our scorer, plus refined hybrids |
| `aggregate/` | scoring of all submissions and the tables |
| `recipes/` | the shell scripts that run a stage end to end, one folder per study |
| `results/` | the final result tables and the tag index |

Each folder has its own `README.md` with every script, its inputs and its outputs.

## Setup

1. Python 3 with the packages in `requirements.txt` (`pip install -r ir_methods/requirements.txt`). It was written on Python 3.14. The CPU baselines need only numpy, scipy, scikit-learn, nltk and snowballstemmer.
2. `source ir_methods/env.sh` in every new shell. It sets the paths below and puts every stage folder on `PYTHONPATH`.
3. `bash ir_methods/setup_work.sh models` downloads the public base models. Skip `models` if you only run CPU recipes.
4. A GPU with about 11 GB is enough for every neural recipe. The `rl_style` scripts import the `method/` package of the repo root, so that folder must be present next to `runs/`.

| Variable | Meaning | Default |
|---|---|---|
| `IRBENCH_RUNS` | folder holding the eight public runs | `<repo>/runs` |
| `IRBENCH_WORK` | work folder: `data/`, `models/`, caches, `out/` | `ir_methods/work` |
| `IRBENCH_OUT` | output folder of the current task, set by each recipe | `$IRBENCH_WORK/out/<task>` |
| `IRBENCH_SCRATCH` | scratch for adapters while training | `$IRBENCH_WORK/scratch` |
| `IRBENCH_KEYS` | set to 1 by diagnostic scripts that score against a key | unset |
| `RUNS` | comma list of runs a Python script works on | all eight |
| `RUNSEL` | space list of runs a recipe processes | all eight |
| `DEVR`, `TESTR` | runs scored on the dev and the test side, an empty value means none | all dev, all test |

Everything that is large is regenerated into `$IRBENCH_WORK`: model weights, LoRA adapters, caches, candidate files, submissions. Nothing of that is in the repo.

## Reproduce

### a. One baseline row on CPU (about a minute)

Run from the repo root. This builds the contextual pieces and BM25 lists of `on_call`, writes the `b3_*` baseline submissions and scores them.

```
export IRBENCH_WORK=/some/folder/with/room
export RUNSEL=on_call DEVR= TESTR=on_call
bash ir_methods/recipes/00_prep_lexical.sh
bash ir_methods/recipes/full3b/42_b3_lex.sh
bash ir_methods/recipes/full3b/44_b3_score.sh
python3 score.py runs/on_call $IRBENCH_WORK/out/full3b_42_b3_lex/b3subs/test/b3_bm25_word/on_call.json
```

The scores of every baseline are printed by `44_b3_score.sh`. The last command scores the tag `b3_bm25_word` with the official `score.py`. The measured output of this exact sequence is in the section Smoke test below.

### b. One reranker, train and score (GPU, hours)

The public reader `ftpub` is a LoRA on Qwen3 Reranker 0.6B. It is trained on public SWE agent traces, never on benchmark data.

```
source ir_methods/env.sh
bash ir_methods/recipes/00_prep_lexical.sh                      # pieces and BM25 lists of the eight runs
bash ir_methods/recipes/test5/10_train_public.sh                # step 1 of that file is ftpub, the rest trains the 4B reader and the students
bash ir_methods/recipes/test5/20_readers.sh                     # first stage z2 plus every reader over the five test runs
bash ir_methods/recipes/test5/60_make_subs_and_score.sh         # frozen submissions and scores
```

To train only ftpub, run step 1 of `10_train_public.sh` (the `convert_public_traces.py`, `prepare_public_runs.py`, `generate_questions.py`, `train_reranker.py` chain). Scoring one reader over one run by hand:

```
cd "$IRBENCH_WORK" && python $IR/rerankers/gpu_score.py online_shop z2 Qwen/Qwen3-Embedding-0.6B "Qwen/Qwen3-Reranker-0.6B,Qwen/Qwen3-Reranker-0.6B+lora=models/ftpub" 200
cd "$IRBENCH_WORK" && python $IR/list_builder/write_submissions.py gpu ftpub z2 "Qwen/Qwen3-Reranker-0.6B+lora=models/ftpub"
```

The first line writes the reader scores to `data/online_shop/gpu_z2.json`. The second runs the frozen list builder and writes `$IRBENCH_OUT/subs/ftpub/<run>.json`.

### c. The tables and the scorer

```
python3 score.py runs/<run> SUBMISSION.json                     # official scorer, one submission
python $IR/aggregate/score_test_subs.py                          # every $IRBENCH_OUT/subs/<tag>/<run>.json -> $IRBENCH_OUT/scores
bash ir_methods/recipes/run_all.sh full2 98 99                  # score and aggregate the full2 study into results_full2.csv
bash ir_methods/recipes/run_all.sh full3 98 99                  # the same for the structure aware study, results_full3.csv
bash ir_methods/recipes/run_all.sh full3b 44 99                 # the baselines, results_full3b.csv
```

Each recipe writes to its own folder `$IRBENCH_WORK/out/<study>_<recipe name>`, so the three `99_aggregate.sh` files leave their tables in `out/full2_99_aggregate`, `out/full3_99_aggregate` and `out/full3b_99_aggregate` and the `.done` markers do not collide. The aggregate steps read the submissions and scores of every earlier recipe folder, so run the study first, `run_all.sh full2` with no range does that.

The result tables the document was built from are in `results/`. `results/tag_index.csv` lists every tag of the document, its table, its entry point and its recipe.

## Studies and recipes

The results come from four batches. Each has a recipe folder and a result file.

| Batch | Recipes | Result file | What it holds |
|---|---|---|---|
| test5 | `recipes/test5/` | `results/results_test5.csv` | the frozen test study: first stage recall, zero shot and public readers, students, speed, algorithmic readers, expansion, family sweeps |
| full2 | `recipes/full2/` and `recipes/rl_style/` | `results/results_full2.csv` | the dev plus test study: first stage and readers, public training size, bigger readers, listwise readers, fast pipeline configs, RL style reruns |
| full3 | `recipes/full3/` | `results/results_full3.csv` | structure prefixes and sectional rerank |
| full3b | `recipes/full3b/` | `results/results_full3b.csv` | baselines with no training on all eight runs |

A tag that appears in more than one batch is kept in the last one, the same rule the document uses. `recipes/run_all.sh <folder> [lo] [hi]` runs the numbered recipes of a folder in order and skips finished ones.

## Which code made which tag

The full list of 330 tags is `results/tag_index.csv`. By family:

| Tags | Doc section | Entry point | Recipe |
|---|---|---|---|
| `b3_*` | B3 baselines | `baselines/b3_lex.py`, `b3_dense.py`, `aggregate/score_submissions.py` | `recipes/full3b/` |
| `example_bm25`, `bm25_ctx_alone` | 4.1 | `example_bm25.py` (repo root), `baselines/bm25_ctx_alone.py` | `recipes/00_prep_lexical.sh`, then `recipes/test5/60_make_subs_and_score.sh` for `bm25_ctx_alone` |
| `fs_*`, `rec200_*`, `time_fuse0` | 6.1, 6.3, 6.4, 6.5 | `first_stage/first_stage_recall.py`, `cpu_families.py`, `family_recall.py`, `write_fuse0.py` | `recipes/test5/30_cpu_families.sh` |
| `zs06`, `zs4b`, `z3`, `c4b`, `bgem3_*`, `ens_zs*`, `lw4` | 5.2 | `rerankers/gpu_score.py`, `cascade.py`, `listwise.py`, `list_builder/write_submissions.py` | `recipes/test5/20_readers.sh`, `60_make_subs_and_score.sh` |
| `ftpub`, `ftpub2x`, `ftpub4b`, `ftpub4b2x`, `ens_ftpub*` | 5.3 | `rerankers/train_reranker.py`, `gpu_score.py`, `list_builder/write_submissions.py` | `recipes/test5/10_train_public.sh`, `20_readers.sh`, `60_make_subs_and_score.sh` |
| `ft4b_*`, `ftsyn_*`, `fttrans`, `ftreal_*`, `ftwho_*` | 5.4 probes | same as above | `recipes/test5/20_readers.sh`, `60_make_subs_and_score.sh` |
| `ce_*`, `ds_bgeb`, `rstul*`, `cas_rstul*` | 5.5, 5.6 | `students/`, `list_builder/write_pool_submissions.py stu` | `recipes/test5/10_train_public.sh`, `40_pool_and_students.sh` |
| `ftpub_base_*`, `ftpub_exit_*`, `fam_*` | 6.4, 6.6 | `first_stage/reader_pools.py`, `family_pool.py`, `rerankers/score_pool.py`, `list_builder/write_pool_submissions.py` | `recipes/test5/40_pool_and_students.sh` |
| `algread_*`, `filelevel_*`, `convprf_*`, `d2q_*` | 6.7 | `algorithmic/`, `expansion/` | `recipes/test5/30_cpu_families.sh` |
| `fast_*`, `time_fast_*`, `t1` | 7 | `fast_pipeline/timed_run.py`, `timed_run_subs.py`, `run_method.py` | `recipes/test5/50_fast_pipeline.sh` |
| `f2_*`, `t2_*` | 9.1 to 9.8 | `rerankers/score_z2.py`, `train_ftpub.py`, `listwise_multi.py`, `score_pool_4b.py`, `fusion.py`, `fast_pipeline/` | `recipes/full2/` |
| `f3_*`, `t3_*` | 9.1 to 10.2 | `structure_aware/` | `recipes/full3/` |
| `pl_*` | S1, S2 | `rl_style/` | `recipes/rl_style/` |

## Things to know before you rerun

- Some adapters and students listed as probes (section 5.4) were trained on dev benchmark records on purpose, to measure what in sample training buys. Their rows say so in `results/results_test5.csv`.
- The exact training command of a few variants (`ftpub2x`, `ftpub4b`, `ftpub4b2x`, `ft4b_*`, `ftsyn_*`, `fttrans`, `ftreal_lib`, `ftwho_lib`, and the students `ce_*` and `rstul*`) was not kept. `recipes/test5/10_train_public.sh` holds the `ftpub` chain that was kept and rebuilds the others from the settings written in the result rows. A second disjoint batch of public traces, as the 2x adapters used, is made with `OFFSET` and `K0` of `convert_public_traces.py`. Treat those as reconstructions.
- The test study used a cluster, so the recipes in `recipes/test5/` are condensed and run on one machine. Hours of GPU time are normal. The timings in the result files were measured on the original hardware and will differ.
- Training is seeded but GPU kernels are not bit exact. Expect small differences in the third decimal.
- No hosted LLM API is called anywhere. Question generation and listwise readers use local models.
- Scripts that are not an entry point of any tag are helpers imported by one that is. Anything no tag in the document uses was left out.

## Smoke test

The sequence in section a was run from an empty environment (`env -i`) on a fresh work folder, CPU only, in about 50 seconds. The output of the four commands, with the time stamp and the 1400 line JSON breakdown of `score.py` left out:

```
on_call 12488 pieces
done test on_call 12488 pieces 50 q 10.3s
LEX_ALL_DONE
test b3_bm25_ident complete on_call=0.387
test b3_bm25_stem complete on_call=0.372
test b3_bm25_stem_lb complete on_call=0.668
test b3_bm25_word complete on_call=0.361
test b3_bm25_word_lb complete on_call=0.617
test b3_bm25_ws complete on_call=0.246
test b3_long complete on_call=0.007
test b3_oldest complete on_call=0.027
test b3_random_s0 complete on_call=0.120
test b3_random_s1 complete on_call=0.076
test b3_random_s2 complete on_call=0.056
test b3_recent complete on_call=0.097
test b3_recent_lb complete on_call=0.099
test b3_short complete on_call=0.000
score 0.361 out of 1 (equal credit 0.390) | every atom found within 2k tokens for 30%, within 32k for 60% of 50 questions
```

`score.py` prints the summary line first and then a JSON breakdown of about 1400 lines, so add `> /dev/null` if you only want the score line. That line is the `b3_bm25_word` submission and matches the `test_on_call` value of that tag in `results/results_full3b.csv`. The GPU recipes were not run on that machine: they need about 11 GB of GPU memory.
