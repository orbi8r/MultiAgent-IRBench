# MultiAgent-IRBench

Test your search method on the memory of an AI agent team.

![How it works](docs/how_it_works.svg)

A team of AI agents works on a real job for hours, and every step they take is saved. Later one of them asks "where did we put X?". Your method searches the saved record and hands back the pieces that answer it. `score.py` grades those pieces.

## What is new

- 3 runs and 146 questions are published for now: `online_shop`, `library_maintainers` and `literature_review`. The other 7 runs are withdrawn for the moment while they are rebuilt.
- The questions are new. They were written from what the agents actually looked up while they worked.
- The answer keys are new and complete. Each key comes from a sweep of the whole record, judged by one frozen Opus procedure. The keys are complete and consistent with that procedure, which was also tested on runs it never saw.
- The result line now has two numbers. The second one counts a piece that states only part of an answer as half a hit.
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
score 0.536 out of 1 (0.510 with half credit) | every part found within 2k tokens for 53%, within 8k for 59% of 49 questions
```

The first number is the main score. The second one gives partial pieces half credit. A method that returns nothing scores 0.

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

![How the score works](docs/scoring.svg)

1. Each question has an answer that the record states in some pieces. The key lists all of them: the first statement, later copies and the raw tool output that shows it.
2. The agent reads your list from the top. Each piece costs its tokens, at least 64. Reading stops at the piece that would cross 8,000 tokens.
3. After 1,000, 2,000, 4,000 and 8,000 tokens, the share of answer parts already stated by a piece you returned is counted. The question scores the mean of these four shares.
4. Some pieces state only part of an answer. The main score counts them. The second number counts that part as half found until a piece that states all of it is read.
5. For who and history questions only the first statement counts, since its author and time are what was asked.
6. The same file always gets the same score. No AI is involved.

Tip: short pieces first. One long event can eat the whole budget.

## Leaderboards

We ran 27 methods on every run with the same code, and no method saw the answer keys. The tables show the top 5. A line marked * was fitted on the old questions of that same run, so its rank there is not a held out result. Methods with the same score returned the same lists.

The oracle is the best list that can be built from the key. Its score is 1.000 on every run, and with half credit it is 1.000 on online_shop, 0.904 on library_maintainers and 0.980 on literature_review. The example method `example_bm25.py` scores 0.536, 0.511 and 0.615 in the same order.

### online_shop

| Rank | Method | Kind | Score | With half credit |
|---|---|---|---|---|
| 1 | `ltr_frozen` | learned to rank | 0.770 | 0.750 |
| 2 | `ltr_fused_frozen` | learned to rank | 0.750 | 0.727 |
| 3 | `mm_frozen` | memory map | 0.724 | 0.696 |
| 4 | `o_rrf_full_ce2_w0.2` | origin reranker | 0.724 | 0.696 |
| 5 | `c_rrf_kind_field_neigh_moves_dedup` | combo | 0.689 | 0.661 |

### library_maintainers

| Rank | Method | Kind | Score | With half credit |
|---|---|---|---|---|
| 1 | `p_bm25_dedup` | structure | 0.654 | 0.614 |
| 2 | `s_dedup` | structure | 0.654 | 0.614 |
| 3 | `h_rrf60_bm25_bge_base_e5_small` | hybrid fusion | 0.644 | 0.620 |
| 4 | `h_ce_bgerr_top100_rrf` | hybrid reranker | 0.638 | 0.617 |
| 5 | `ltr_frozen` * | learned to rank | 0.638 | 0.596 |

### literature_review

| Rank | Method | Kind | Score | With half credit |
|---|---|---|---|---|
| 1 | `ltr_frozen` | learned to rank | 0.720 | 0.700 |
| 2 | `ltr_fused_frozen` | learned to rank | 0.680 | 0.655 |
| 3 | `p_bm25_dedup` | structure | 0.670 | 0.650 |
| 4 | `s_dedup` | structure | 0.670 | 0.650 |
| 5 | `h_ce_bgerr_top100_rrf` | hybrid reranker | 0.650 | 0.627 |

## The runs

| Run | The team | Questions | Size |
|---|---|---|---|
| `online_shop` | 3 staff and 2 helpers run a simulated shop on a real UK retailer's orders: stock, prices, books and monthly reports | 49 | 1.4M tokens |
| `library_maintainers` | 10 agents ship a release of the dateutil library from its real issue list | 47 | 1.0M |
| `literature_review` | 5 researchers write literature reviews from real arXiv papers | 50 | 1.2M |

146 questions in all. The other 7 runs are withdrawn for now.
