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
| Reading | best value per part inside each budget, normalised | share of parts stated after reading 1k, 2k, 4k and 8k tokens, averaged |
| Later copy of a fact | 0.4 | counts fully (except who and history questions) |
| Raw tool output | 0.5 | counts fully (except who and history questions) |
| Who and history questions | origin 1.0, copies less | only the first statement counts |
| Piece too big for the budget | skipped, reading continues | reading stops there: the agent reads in order |
| Partial pieces | not marked | counted in the main score, counted half in the second number |
| Minimum cost per piece | 64 tokens | 64 tokens |

The report now has two scores. `overall.score` counts a partial piece as a found part. `half_credit.overall.score` counts that part as half found until a piece that states all of it is read. Use the first one to compare with others and the second one to see how much of your result rests on partial pieces.

The old scorer is kept as `score_v1.py`. It reproduces the old numbers only with the old answer keys, which are in the git history of this repo.

## Report fields

- `overall.score`: the score, from 0 to 1.
- `overall.answered_within_1000` to `_8000`: the share of questions with every part found within that many tokens.
- `per_query[qid]`: `score`, `shares` (one per budget) and `answered_at` (tokens read until every part was found, or null).
- `by_qtype`: the same summary per question type.
- `half_credit`: the same fields again, with partial pieces counted half. It has its own `overall`, `by_qtype` and `per_query`.
- Gone: `above_floor`, `full_support` and `by_budget`. They are still in `score_v1.py`.

## If you train with the score as a reward

`score_query` has the same name and signature. `score_query_half` is new and takes the same arguments.

```python
from score import score_query, score_query_half, load_tokens
tokens = load_tokens("runs/x/events.jsonl", "runs/x/blocks.jsonl")
reward = score_query(key[qid], ranked_ids, tokens)["score"]
reward_half = score_query_half(key[qid], ranked_ids, tokens)["score"]
```

- The reward moves in steps: a part counts once it is read, at 1k, 2k, 4k or 8k tokens. If that is too sparse early on, use `shares` (four numbers per question) as a denser signal.
- Copies count, so a model that learned to push origins above copies is not penalised. A model that only learned copies will score well, except on who and history questions.
- The questions and keys changed, so a checkpoint trained on the old keys needs fresh training or at least a new evaluation on these runs.
