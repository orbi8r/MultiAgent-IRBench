"""Render a results csv as a short markdown table: tag, method, per run dev and test scores and their means.

Usage: python3 aggregate/csv_to_md.py results/results_full2.csv > results/results_full2.md
"""
import csv
import sys

COLS = ["tag", "method", "dev_online", "dev_library", "dev_literature", "dev_mean",
        "test_on_call", "test_due_diligence", "test_executive_assistants", "test_help_desk",
        "test_textbook_team", "test_mean"]


def main(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print("| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for r in rows:
        cells = [(r.get(c) or "").replace("|", "/").replace("\n", " ") for c in COLS]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
