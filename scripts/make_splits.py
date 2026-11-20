"""Document-level train/dev/test split of the processed corpus (PRD FR-H2).

Whole documents go to one split, so no book, newspaper issue or Wikipedia
shard is shared between train and test. Within each category documents are
shuffled with a fixed seed; dev and test each get round(10%) of the documents
(at least 1 when the category has >= 3 documents), the rest is train.

Outputs in --out:
    train.txt, dev.txt, test.txt   one sentence per line
    split_manifest.csv            doc_id, category, split, sentences, tokens

Usage:
    python scripts/make_splits.py --processed data/processed --out data/splits
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from corpus_utils import CATEGORIES

SEED = 2026


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--dev", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    docs = defaultdict(list)                     # doc_id -> [(sentence, n_tokens)]
    cat_of = {}
    for cat in CATEGORIES:
        tsv = Path(args.processed) / "by_category" / cat / "sentences.tsv"
        if not tsv.exists():
            continue
        with open(tsv, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                docs[row["doc_id"]].append((row["sentence"], int(row["n_tokens"])))
                cat_of[row["doc_id"]] = cat

    split_of = {}
    rng = random.Random(SEED)
    for cat in CATEGORIES:
        ids = sorted(d for d, c in cat_of.items() if c == cat)
        rng.shuffle(ids)
        n = len(ids)
        n_dev = max(1, round(args.dev * n)) if n >= 3 else 0
        n_test = max(1, round(args.test * n)) if n >= 3 else 0
        for i, d in enumerate(ids):
            split_of[d] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"

    handles = {s: open(out / f"{s}.txt", "w", encoding="utf-8", newline="\n") for s in ("train", "dev", "test")}
    totals = defaultdict(lambda: [0, 0, 0])
    with open(out / "split_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["doc_id", "category", "split", "sentences", "tokens"])
        for d in sorted(docs, key=lambda x: (cat_of[x], x)):
            s = split_of[d]
            ntok = sum(n for _, n in docs[d])
            for sent, _ in docs[d]:
                handles[s].write(sent + "\n")
            w.writerow([d, cat_of[d], s, len(docs[d]), ntok])
            t = totals[s]
            t[0] += 1
            t[1] += len(docs[d])
            t[2] += ntok
    for h in handles.values():
        h.close()
    grand = sum(t[2] for t in totals.values())
    for s in ("train", "dev", "test"):
        docs_n, sents, toks = totals[s]
        print(f"{s:5s} {docs_n:4d} docs {sents:>9,d} sentences {toks:>10,d} tokens ({100 * toks / grand:.1f}%)")


if __name__ == "__main__":
    main()
