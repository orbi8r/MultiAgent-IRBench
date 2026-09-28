"""A tiny example method: rank blocks by BM25 for every question and write a submission file.

    python example_bm25.py runs/debate_panel my_submission.json

Plain Python, no packages. Replace the ranking with your own method; keep the output format.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def words(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def main(run_dir, out_path):
    run = Path(run_dir)
    blocks = [json.loads(line) for line in open(run / "blocks.jsonl", encoding="utf-8")]
    questions = [json.loads(line) for line in open(run / "questions.jsonl", encoding="utf-8")]
    docs = [Counter(words(b["text"])) for b in blocks]
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / len(lengths)
    df = Counter(w for d in docs for w in d)
    postings = defaultdict(list)
    for i, d in enumerate(docs):
        for w in d:
            postings[w].append(i)
    n = len(docs)
    submission = {}
    for q in questions:
        scores = defaultdict(float)
        for w in set(words(q["question"])):
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            for i in postings.get(w, ()):
                tf = docs[i][w]
                scores[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[i] / avg))
        best = sorted(scores, key=lambda i: -scores[i])[:300]
        submission[q["qid"]] = [blocks[i]["id"] for i in best]
    json.dump(submission, open(out_path, "w", encoding="utf-8"))
    print(f"wrote {len(submission)} ranked lists to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
