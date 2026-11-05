# rl_style

Re run of an existing reinforcement learning style reranking pipeline (Model A, a neural policy trained with Plackett Luce REINFORCE, and Model B, a linear policy with coordinate ascent) under the protocol of this study. Tags `pl_*`, doc section on the RL style methods.

Protocol: fit on the three dev runs only, leave one run out for the dev numbers, all three dev runs for the frozen test setting. No evaluation based checkpoint selection. Only dev answer keys are opened, and `rl_common.py` asserts it.

These scripts import the `method/` package of the repo (the original pipeline modules), so run them from the repo root, not from `$IRBENCH_WORK`. The recipes in `recipes/rl_style/` do that with `cd "$IRBENCH_REPO"`. The `method/` folder is not part of this folder and is not changed.

| Script | What it does | Writes |
|---|---|---|
| `rl_common.py` | shared helpers, key guard, frozen files, rows, timing | imported |
| `features.py OUT` | key free stage: first stage list and the candidate features the original policies rank | `$RL_STYLE_FEATS/` |
| `fit_rank.py OUT` | first stage lists, rules only, Model B fitted on dev and the shipped checkpoint probe | subs, frozen, rows, devscore, weights, timing under `OUT` |
| `fit_model_a.py OUT` | Model A under the declared settings: 0 imitation epochs, 15 RL epochs, 16 samples, seed 0, last epoch | the same |
| `patched_rl_trainer.py` | the original `method/rl_trainer.py` with one line changed, see below | imported by `fit_model_a.py` |
| `pool_features.py OUT` | the original features for our pool (fuse0 plus expansion) and the ftpub score of every pool piece | `$RL_STYLE_FEATS/pool_<run>.pkl` |
| `hybrid_lib.py` | the original coordinate ascent generalised to any feature matrix, so the ftpub score can be a 13th feature | imported |
| `hybrid_fit.py` | the refined hybrids h1 to h4 on our pool, fitted on the dev runs | subs, rows, frozen |
| `check_coordinate_ascent.py [n_iter]` | check that `hybrid_lib` on the original 12 features gives the same weights as the original fit | prints the comparison |

The trainer. `patched_rl_trainer.py` holds no copy of the teammate's `method/rl_trainer.py`. It reads that file, replaces one line in `train_epoch` and runs the result as the module `patched_rl_trainer`, so `from patched_rl_trainer import RLTrainer` is the original class. The line is the reward call: `score_query(answer_entry, ranked_ids, tokens_map)` becomes `score_query(answer_entry, ranked_ids, item["run"], credit=item["credits"])`, because the `score.py` of this repo takes the loaded run and the credit table of the question instead of a token map, and `fit_model_a.py` puts both into every item. Nothing else differs. `RLTrainer.evaluate` keeps the old call and would fail under this scorer, but `fit_model_a.py` never calls it (it trains with no eval data and no imitation epochs). If `method/rl_trainer.py` changes, the patch stops with an assertion instead of running a different trainer.

Recipes in order: `70_pl_feats.sh`, `71_pl_fit.sh`, `72_pl_modelA.sh`, then `21_pl_pool.sh` and `22_pl_hybrid_fit.sh` after the ftpub pool step of `recipes/full2/20_ftpub_pool.sh`.
