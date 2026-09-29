"""Start here. Put your search method in rank(), then:

    python my_method.py runs/on_call my_submission.json
    python score.py runs/on_call my_submission.json > report.json
"""
import json
import sys
from pathlib import Path


def rank(question, blocks):
    """question: {"qid", "question", "qtype", "asker"}. blocks: list of {"id", "event", "tokens", "text"}.
    Return block ids (or event ids), best first. This placeholder counts shared words."""
    words = set(question["question"].lower().split())
    hits = [(len(words & set(b["text"].lower().split())), b["id"]) for b in blocks]
    return [i for n, i in sorted(hits, reverse=True)[:200] if n]


def main(run_dir, out_path):
    run = Path(run_dir)
    blocks = [json.loads(line) for line in open(run / "blocks.jsonl", encoding="utf-8")]
    questions = [json.loads(line) for line in open(run / "questions.jsonl", encoding="utf-8")]
    json.dump({q["qid"]: rank(q, blocks) for q in questions}, open(out_path, "w", encoding="utf-8"))
    print(f"wrote {len(questions)} ranked lists to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
