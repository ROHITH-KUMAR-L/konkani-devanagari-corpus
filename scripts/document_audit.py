"""Per-document token counts through every cleaning iteration -> results/document_audit.csv.

Shows how much of each source survives each step, so a source that loses an
unusual share of its text (wrong language, OCR garbage, duplicates) stands out.

Usage:
    python scripts/document_audit.py --iters data/interim/iter0 data/interim/iter1 \
        data/interim/iter2 data/interim/iter3 --out results/document_audit.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from corpus_utils import iter_docs, read_doc, tokenize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", nargs="+", required=True)
    ap.add_argument("--out", default="results/document_audit.csv")
    args = ap.parse_args()
    counts, cat_of = defaultdict(dict), {}
    names = [Path(p).name for p in args.iters]
    for name, root in zip(names, args.iters):
        for cat, doc_id, path in iter_docs(root):
            counts[doc_id][name] = len(tokenize(read_doc(path)))
            cat_of[doc_id] = cat
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["doc_id", "category"] + [f"tokens_{n}" for n in names] + ["kept_pct"])
        for d in sorted(counts, key=lambda x: (cat_of[x], x)):
            row = [counts[d].get(n, 0) for n in names]
            w.writerow([d, cat_of[d]] + row + [f"{100 * row[-1] / row[0]:.1f}" if row[0] else "0.0"])
    print(f"{len(counts)} documents -> {args.out}")


if __name__ == "__main__":
    main()
