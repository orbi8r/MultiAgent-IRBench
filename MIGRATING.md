# Moving to the new runs and the new scorer

Both changed. The questions and answer keys of the runs are new, and the score is computed a new way. The record files (`events.jsonl`, `blocks.jsonl`) are the same as before.

## What to do

1. Pull the repo.
2. Run the same command as before: `python score.py runs/<name> my_submission.json > report.json`
3. Run your method again on the 3 runs now in `runs/`. Old submission files cannot be rescored, because the question ids differ from before.

## What changed in the runs

- Only 3 runs are published for now: `online_shop`, `library_maintainers` and `literature_review`, 146 questions in all. The other 7 runs are withdrawn while they are rebuilt.
- The questions are new. They were written from what the agents actually looked up while they worked.
- The answer keys are new and complete. Every part of an answer lists all the pieces of the record that state it, found by sweeping the whole record.
- Some pieces state only part of an answer part. The key marks them, and they are the reason for the second number below.

## What changed in the score

| | Old | New |
|---|---|---|
| Reading | best value per part inside each budget, normalised | share of atoms found after 125, 250, 500, 1k, 2k, 4k, 8k, 16k and 32k tokens, averaged |
| Reading stops | 8,000 tokens | 32,000 tokens |
| Answer split into atoms | no | yes: each separate fact an answer needs is an atom, and a piece earns only the atoms it states |
| First statement of a fact | 1 | 1 |
| Later copy of a fact | 0.4 | 1 if it states the atom: identical text or a copy the key lists counts like the first statement |
| Raw tool output | 0.5 | 1 for each atom it states |
| Who and history questions | origin 1.0, copies less | a copy counts only if its text or its event's agent and shift carry the asked ones |
| Piece too big for the budget | skipped, reading continues | reading stops there: the agent reads in order |
| Partial pieces | not marked | earn the atoms they state, nothing more |
| Scale | normalised by the best reachable value | none: the same formula and constants on every run, a budget is never below the cost of the question's perfect answer, so a perfect list scores 1 |
| Cost per piece | its tokens, at least 64 | its tokens plus 16 |

The report has two scores. `overall.score` is the main score. `equal_credit.overall.score` is the rule of the first release of these runs, unchanged: budgets 1k to 8k, any piece that states any atom finds the whole answer and a copy counts like the first statement. Use the first one to compare with others. The gap between the two shows how much of a result rests on copies, partial pieces and late finds.

Why it changed. Under the old rule a piece that stated one fact of a two fact answer counted as the whole answer, each question was close to pass or fail inside 8,000 tokens, and a blind audit found lists that served the agent better but scored lower. The new rule credits each fact separately, reads deeper, rewards an early find over a late one, and passed about 30,000 adversarial checks with no case of a better list scoring lower. Over all three runs 55% of method pairs are told apart against 47% before, and the best method of each family ranks far more alike from run to run (Spearman 0.34 against 0.17). Within one run of about 50 questions the share stays between 30% and 48%: there are not enough questions for finer calls, so compare methods by the mean over the three runs.

The scale is close to the equal credit rule: the typical method built without the keys scores about 0.5 on every run. Methods fitted to the old keys of these runs gain at most 0.02 over the best fair method. An empty list scores 0, methods that ignore the question score 0.03 to 0.06 and the oracle scores 1 on every run.

The old scorer is kept as `score_v1.py`. It reproduces the old numbers only with the old answer keys, which are in the git history of this repo.

## Report fields

- `overall.score`: the score, from 0 to 1.
- `overall.answered_within_125` to `_32000`: the share of questions with every atom stated within that many tokens (or the ideal cost, if larger).
- `per_query[qid]`: `score`, `shares` (one per budget) and `answered_at` (tokens read until every atom was stated, or null).
- `by_qtype`: the same summary per question type. Location questions are the hardest for every method, so read them apart.
- `equal_credit`: the same fields under the equal credit rule, with budgets 1k to 8k. It has its own `overall`, `by_qtype` and `per_query`.
- Gone: `overall.raw`, `half_credit`, `normalise` and the per run exponents. There is no scale any more.
- Gone: `above_floor`, `full_support` and `by_budget`. They are still in `score_v1.py`.

## If you train with the score as a reward

`score_query(entry, ranked_ids, run)` now takes the loaded run instead of a token table, because credit needs the text, agent and shift of each piece. `load_run` loads it once.

```python
from score import score_query, score_query_equal, load_run
run = load_run("runs/x")
reward = score_query(key[qid], ranked_ids, run)["score"]
reward_equal = score_query_equal(key[qid], ranked_ids, run)["score"]
```

For speed, compute `credits(key[qid], run)` once per question and pass it as `credit=`.

- The reward moves in steps: an atom counts once it is read, at nine budgets from 125 to 32,000 tokens. That is denser than before. If it is still too sparse, use `shares` (nine numbers per question).
- Any piece that states an atom earns it, so the reward pushes cheap pieces that state the answer to the top. On who and history questions a copy earns only if it carries the asked agent and shift.
- There is no scale any more: the run score is the plain mean of the question scores, so the mean of per question rewards equals the headline.
- The questions and keys changed, so a checkpoint trained on the old keys needs fresh training or at least a new evaluation on these runs.
