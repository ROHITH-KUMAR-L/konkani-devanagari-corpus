"""Document-level train/dev/test split of the processed corpus (PRD FR-H2).

Whole documents go to one split, so no book, newspaper issue or Wikipedia
shard is shared between train and test. Within each category documents are
shuffled with a fixed seed; dev and test each get round(10%) of the documents
(at least 1 when the category has >= 3 documents), the rest is train.

A domain-balanced copy (PRD FR-C6: no category above --max-share of tokens)
is written to <out>/balanced/: for any category above the cap, documents are
kept at evenly spaced positions in doc_id order (for newspapers that spreads
them over the whole date range) until the category is at the cap. Documents
keep the split they have in the full corpus.

Outputs in --out:
    train.txt, dev.txt, test.txt   one sentence per line (full corpus)
    balanced/{train,dev,test}.txt  same, domain-balanced subset
    split_manifest.csv            doc_id, category, split, sentences, tokens, in_balanced
    results/split_summary.json    totals per split and for the balanced subset (--summary)

Usage:
    python scripts/make_splits.py --processed data/processed --out data/splits
"""
import argparse
import csv
import json
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
    ap.add_argument("--max-share", type=float, default=0.6)
    ap.add_argument("--summary", default="results/split_summary.json")
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

    balanced = select_balanced(docs, cat_of, args.max_share)
    (out / "balanced").mkdir(exist_ok=True)
    handles = {s: open(out / f"{s}.txt", "w", encoding="utf-8", newline="\n") for s in ("train", "dev", "test")}
    bal_handles = {s: open(out / "balanced" / f"{s}.txt", "w", encoding="utf-8", newline="\n")
                   for s in ("train", "dev", "test")}
    totals = defaultdict(lambda: [0, 0, 0])
    with open(out / "split_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["doc_id", "category", "split", "sentences", "tokens", "in_balanced"])
        for d in sorted(docs, key=lambda x: (cat_of[x], x)):
            s = split_of[d]
            ntok = sum(n for _, n in docs[d])
            for sent, _ in docs[d]:
                handles[s].write(sent + "\n")
                if d in balanced:
                    bal_handles[s].write(sent + "\n")
            w.writerow([d, cat_of[d], s, len(docs[d]), ntok, int(d in balanced)])
            t = totals[s]
            t[0] += 1
            t[1] += len(docs[d])
            t[2] += ntok
    for h in list(handles.values()) + list(bal_handles.values()):
        h.close()
    grand = sum(t[2] for t in totals.values())
    for s in ("train", "dev", "test"):
        docs_n, sents, toks = totals[s]
        print(f"{s:5s} {docs_n:4d} docs {sents:>9,d} sentences {toks:>10,d} tokens ({100 * toks / grand:.1f}%)")
    bal_tok = defaultdict(int)
    for d in balanced:
        bal_tok[cat_of[d]] += sum(n for _, n in docs[d])
    bt = sum(bal_tok.values())
    print(f"balanced subset (max share {args.max_share:.0%}): {len(balanced)} docs, {bt:,} tokens; "
          + ", ".join(f"{c} {100 * t / bt:.1f}%" for c, t in sorted(bal_tok.items())))
    summary = {"seed": SEED, "max_share": args.max_share,
               "splits": {s: dict(zip(["documents", "sentences", "tokens"], totals[s])) for s in ("train", "dev", "test")},
               "balanced": {"documents": len(balanced), "tokens": bt, "category_tokens": dict(sorted(bal_tok.items()))}}
    Path(args.summary).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


def select_balanced(docs, cat_of, max_share):
    """Doc ids of a subset in which no category exceeds max_share of the tokens."""
    tok = {d: sum(n for _, n in v) for d, v in docs.items()}
    by_cat = defaultdict(list)
    for d in sorted(docs):
        by_cat[cat_of[d]].append(d)
    keep = set(docs)
    changed = True
    while changed:                      # capping one category shrinks the others' base; repeat until stable
        changed = False
        cat_tok = {c: sum(tok[d] for d in ids if d in keep) for c, ids in by_cat.items()}
        for c, ids in by_cat.items():
            others = sum(t for k, t in cat_tok.items() if k != c)
            cap = max_share / (1 - max_share) * others
            if cat_tok[c] <= cap:
                continue
            # evenly spaced documents: the largest such subset that fits under the cap
            for n in range(len(ids), 0, -1):
                step = len(ids) / n
                chosen = [ids[int(i * step)] for i in range(n)]
                if sum(tok[d] for d in chosen) <= cap:
                    break
            keep -= set(ids) - set(chosen)
            changed = True
            break
    return keep


if __name__ == "__main__":
    main()
