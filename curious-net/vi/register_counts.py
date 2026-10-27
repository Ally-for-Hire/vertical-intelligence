"""Counts of pre-registered predictions and their verdicts, straight from paper/PREDICTIONS.md.
Used for the 'at a glance' line of the public README, so the numbers are never typed by hand.
Usage: python register_counts.py     -> prints e.g.: N pre-registered predictions, A passed, B failed, C awaiting results
Resolution-rule rows (ids like P37-res) are not predictions and are not counted."""
import os, re
from collections import Counter

REG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "PREDICTIONS.md")


def counts():
    rows = [l for l in open(REG, encoding="utf-8") if re.match(r"\| P\d+\w* \|", l)]
    c = Counter("passed" if "PASSED" in l else "failed" if "FAILED" in l else "pending" if "pending" in l else "other" for l in rows)
    return len(rows), c


if __name__ == "__main__":
    n, c = counts()
    print(f"{n} pre-registered predictions, {c['passed']} passed, {c['failed']} failed, {c['pending']} awaiting results"
          + (f", {c['other']} other (resolution rules or voided)" if c["other"] else ""))
