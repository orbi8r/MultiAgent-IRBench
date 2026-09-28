# MultiAgentBench

A test for search methods that look things up in the memory of an AI agent team.

Each run is the full record of a team of AI agents working for a long time on a real job, plus 42 to 50 questions that an agent on that team would ask about the team's past ("where did we file the report on X", "what did the lead decide about Y", "which number did the paper give for Z"). Your method reads the record and, for each question, hands back the pieces of the record it would give to that agent. The scorer tells you how good those pieces are.

Three runs are here now. Seven more are being made and will be added to this repository.

## What you need

Python 3.8 or newer, on any operating system. No packages to install.

## What is in a run

Each folder in runs/ is one run:

- events.jsonl: the record, one event per line. Every event has an id, the agent that produced it, its kind (prompt, model, tool_call or tool_result), a time, a shift number, a conversation id, its text and its length in tokens (counted with the cl100k_base tokenizer). This is what your method searches.
- blocks.jsonl: the same record cut into small pieces of up to 8 lines. Every block has an id like "n000123/9f3c2a1b7d4e6f80", the event it belongs to, its text and its length in tokens. You may return whole events or single blocks, whichever your method prefers. Some very short events have no block; return those as whole events.
- questions.jsonl: the questions, one per line, with a qid, the question text, a type (location, state, finding, decision, history, who, failure_fix, superseded or open) and, where the question speaks as "I" or "my", the asker: the agent in the record who is asking.
- answer_key.json: the answer key the scorer uses. Your method must not read it.
- empty_submission.json: the shape of a submission, with every question and an empty list.

The runs:

- library_maintainers: a team of 12 agents (10 of them appear in the record) prepares a release of a real open source library (dateutil) from its real issue backlog. 42 questions, about 1M tokens.
- debate_panel: 6 agents argue real disputed physics questions for a workshop from real research papers, one session after another. 50 questions, about 1.8M tokens.
- textbook_team: a team of 11 agents in three levels (8 of them appear in the record) writes a question bank with worked solutions from real open textbooks. 50 questions, about 1.1M tokens.

## What you return

One JSON file per run: for every qid, a list of event ids or block ids, best first.

    {"b07": ["n000812/1a2b3c4d5e6f7a8b", "n000815", "..."], "b08": ["..."]}

Return as many as you like. The scorer reads your list in order and keeps what fits into reading budgets of 1,000, 2,000, 4,000 and 8,000 tokens (every item costs at least 64 tokens), so the order matters and dumping everything does not help.

## Getting a score

    python score.py runs/debate_panel/answer_key.json my_submission.json --events runs/debate_panel/events.jsonl --blocks runs/debate_panel/blocks.jsonl

It prints a one line summary, and the full report (per question, and per type of question) as JSON. It stops with a message if your file names ids it cannot find, so pass both the events and the blocks file of the same run. The numbers:

- score, from 0 to 1: how much of each answer your pieces carry, compared with the best any list could do under the same budget, averaged over the questions. Handing back the piece where the team first worked out the answer earns full credit; the raw material it came from, later copies and the lookup that fetched it earn less; everything else earns nothing.
- above the no reading floor: your score minus the best score a method gets without reading the questions at all. This is the number to compare.
- full support: the share of questions where your pieces cover every part of the answer.

The same submission always gets the same score, and no AI model is involved in scoring.

## An example

example_bm25.py is a small keyword search method that writes a valid submission:

    python example_bm25.py runs/debate_panel my_submission.json

It scores 0.31 to 0.35 on these runs. Replace its ranking with your own method and keep the output format.
