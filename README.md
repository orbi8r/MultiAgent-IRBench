# MultiAgent-IRBench

Test your search method on the memory of an AI agent team.

![How it works](docs/how_it_works.svg)

A team of AI agents works on a real job for hours, and every step they take is saved. Later one of them asks "where did we put X?". Your method searches the saved record and hands back the pieces that answer it. `score.py` grades those pieces.

## What is new

- 3 runs and 146 questions are published for now: `online_shop`, `library_maintainers` and `literature_review`. The other 7 runs are withdrawn for the moment while they are rebuilt.
- The questions are new. They were written from what the agents actually looked up while they worked.
- The answer keys are new and complete. Each key comes from a sweep of the whole record, judged by one frozen Opus procedure. The keys are complete and consistent with that procedure, which was also tested on runs it never saw.
- The score now splits each answer into atoms, the separate facts it needs, and credits a piece for the atoms it actually states. Reading goes on to 32,000 tokens and an early find is worth more than a late one. A list that serves the agent better never scores lower, a perfect list scores 1 and the same formula holds on every run, with nothing fitted.
- Coming from the earlier version? Read `MIGRATING.md`.

## Try it (1 minute)

You need Python 3.8 or newer. Nothing to install. On Windows type `python`; on Mac or Linux type `python3`.

```
git clone https://github.com/orbi8r/MultiAgent-IRBench
cd MultiAgent-IRBench
python example_bm25.py runs/online_shop my_submission.json
python score.py runs/online_shop my_submission.json > report.json
```

The last line printed is your result:

```
score 0.282 out of 1 (equal credit 0.536) | every atom found in its first statement within 2k tokens for 18%, within 32k for 33% of 49 questions
```

The first number is the main score. Equal credit is the earlier rule, where any piece that states any part of the answer counts fully. A method that returns nothing scores 0 and the best possible list scores 1.

## Plug in your own method

1. Open `my_method.py`.
2. Replace the inside of `rank(question, blocks)` with your search. Return ids, best first.
3. Run and score it:
   ```
   python my_method.py runs/online_shop my_submission.json
   python score.py runs/online_shop my_submission.json > report.json
   ```
4. Do the same for every folder in `runs/`.

Any language or tool works, as long as it writes the same JSON file.

## What is in a run

Each folder in `runs/` is one team.

| File | What it is | Use it |
|---|---|---|
| `questions.jsonl` | the questions | yes |
| `blocks.jsonl` | the record cut into small pieces, up to 8 lines each (usually about 60 tokens) | yes, search these |
| `events.jsonl` | the record as whole steps, in time order (a few tokens up to thousands) | for more context |
| `answer_key.json` | the answers | never, only the scorer reads it |
| `empty_submission.json` | every question with an empty list | to see the output shape |

One line of each:

```
questions.jsonl  {"qid": "g01", "question": "What median price elasticity came out of the per-SKU regressions on the historical demand data?", "qtype": "finding", "asker": "pricing_analyst"}
blocks.jsonl     {"id": "n000007/dbdcf818975b492e", "event": "n000007", "tokens": 19, "text": "STATUS.md (111 bytes)\nspawn/\nto_bookkeeper/\nto_clerk/\nto_manager/"}
events.jsonl     {"id": "n000003", "agent": "manager", "kind": "model", "ts": 1790621529.653, "shift": 1, "conv": "00286400-8929-4a57-bf54-8a25cc52fc2a", "text": "I'll start by getting oriented: reading my notebook, the board status, and my notes.", "tokens": 19}
```

`kind` says what a step is: `prompt` (the agent's instructions at the start of a shift), `model` (what the agent said), `tool_call` (a command it ran or a file it wrote) or `tool_result` (what came back). `asker` is the agent asking, so you know who "I" or "my" means. A few very short events have no block; return those as whole events.

## What you hand back

One JSON file per run. For each question, a list of block ids or event ids, best first:

```
{"g01": ["n000007/dbdcf818975b492e", "n000003", "..."], "g02": ["..."]}
```

The list can be any length.

## How the score works

1. Each question has an answer made of one or more atoms, the separate facts it needs. The key lists, for every atom, each piece that states it: the first statement, later copies and the raw tool output that shows it. The scorer also counts any piece whose text equals a listed piece, or holds a listed text of five words or more whole.
2. The agent reads your list from the top. Each piece costs its tokens plus 16. A repeat is free. Reading stops past 32,000 tokens.
3. A piece earns 1 for every atom it states and 0 for the rest, so one that states half the answer earns half. Where it sits does not matter: an identical or cheaper copy serves the agent as well as the first statement.
4. Who and history questions ask for an author and a time. There a piece other than the first statement counts only if its text, or its event's agent and shift, carry the asked ones.
5. The ideal cost of a question is the cheapest set of pieces that states every atom. After 125, 250, 500, 1,000, 2,000, 4,000, 8,000, 16,000 and 32,000 tokens, each raised to at least the ideal cost, the question takes the share of its atoms read so far. It scores the mean of these nine shares, so an answer found at 300 tokens is worth more than one found at 3,000 or 30,000, and a list that states every atom within the ideal cost scores 1.
6. The run scores the mean over its questions. Every run uses the same formula and the same constants, no term depends on how many questions a run has, the same file always gets the same score and no AI is involved.

Tip: short pieces first. One long event can eat the whole budget.

## Leaderboards

We ran 312 methods on every run with the same code, and no method saw the answer keys. The tables show the top 5. Learned to rank and origin lines were fitted on the old questions of these runs, so their rank there is not a held out result. With about 50 questions a run, two methods less than about 0.05 apart are usually not reliably ordered on one run, so compare methods by the mean over the three runs. An empty list scores 0, methods that ignore the question score 0.03 to 0.06 and the oracle scores 1.000 on every run. The example method `example_bm25.py` scores 0.501, 0.488 and 0.509 on online_shop, library_maintainers and literature_review (equal credit 0.536, 0.511 and 0.615).

### online_shop

| Rank | Method | Kind | Score | Equal credit |
|---|---|---|---|---|
| 1 | ltr_frozen | learned to rank | 0.672 | 0.770 |
| 2 | ltr_fused_frozen | learned to rank | 0.662 | 0.750 |
| 3 | o_rrf_full_cebgerr_w0.5 | origin reranker | 0.655 | 0.735 |
| 4 | o_rrf_full_cebgerr_w0.8 | origin reranker | 0.650 | 0.730 |
| 5 | c_ce_kind_field_neigh_moves_dedup | combo reranker | 0.650 | 0.724 |

### library_maintainers

| Rank | Method | Kind | Score | Equal credit |
|---|---|---|---|---|
| 1 | h_ce_bgerr_top100_rrf | hybrid reranker | 0.639 | 0.638 |
| 2 | h_ce_bgerr_top100_rrf_mix0.5 | hybrid reranker | 0.625 | 0.644 |
| 3 | h_ce_bgerr_top50_rrf | hybrid reranker | 0.625 | 0.654 |
| 4 | p_rrf_dedup_top15_pertoken | packing | 0.622 | 0.681 |
| 5 | p_rrf_dedup | packing | 0.622 | 0.681 |

### literature_review

| Rank | Method | Kind | Score | Equal credit |
|---|---|---|---|---|
| 1 | s_follow_moves_k20_g1.1_loc | structure | 0.622 | 0.730 |
| 2 | c_ce_kind_moves_dedup | combo reranker | 0.619 | 0.720 |
| 3 | c_ce_kind_field_neigh_moves_dedup | combo reranker | 0.597 | 0.665 |
| 4 | ltr_frozen | learned to rank | 0.596 | 0.720 |
| 5 | o_rrf_origin_full | origin | 0.576 | 0.645 |

## The runs

| Run | The team | Questions | Size |
|---|---|---|---|
| `online_shop` | 3 staff and 2 helpers run a simulated shop on a real UK retailer's orders: stock, prices, books and monthly reports | 49 | 1.4M tokens |
| `library_maintainers` | 10 agents ship a release of the dateutil library from its real issue list | 47 | 1.0M |
| `literature_review` | 5 researchers write literature reviews from real arXiv papers | 50 | 1.2M |

146 questions in all. The other 7 runs are withdrawn for now.
