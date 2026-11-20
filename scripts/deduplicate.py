"""Cleaning iteration 3: remove duplicate paragraphs and near-duplicate documents.

Input/output are document trees with one paragraph per line.

  1. Exact paragraph duplicates (PRD FR-N7): paragraphs are compared on their
     letters only (punctuation, digits and spaces ignored), across the whole
     corpus. The first occurrence (category order, then doc_id) is kept. This
     removes repeated newspaper boilerplate and template-generated Wikipedia
     village stubs.
  2. Near-duplicate documents: MinHash (128 hashes) over word 5-gram shingles;
     a document whose estimated Jaccard similarity with an earlier kept
     document is >= --threshold is dropped.

Writes one row per document to metadata/dedup_log.csv.

Usage:
    python scripts/deduplicate.py --in data/interim/iter2 --out data/interim/iter3
"""
import argparse
import csv
import hashlib
import re
import zlib

import numpy as np

from corpus_utils import iter_docs, read_doc, tokenize, write_doc

NUM_HASHES = 128
SHINGLE = 5
PRIME = 4294967311                                    # > 2^32
rng = np.random.default_rng(12345)
A = rng.integers(1, 2 ** 31, NUM_HASHES, dtype=np.uint64)
B = rng.integers(0, 2 ** 31, NUM_HASHES, dtype=np.uint64)


def para_key(p):
    letters = re.sub(r"[\W\d_]+", "", p)
    return hashlib.md5(letters.encode("utf-8")).digest() if letters else None


def minhash(tokens):
    if len(tokens) < SHINGLE:
        return None
    sh = {" ".join(tokens[i:i + SHINGLE]) for i in range(len(tokens) - SHINGLE + 1)}
    x = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))
    return ((np.outer(A, x) + B[:, None]) % PRIME).min(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--log", default="metadata/dedup_log.csv")
    args = ap.parse_args()

    seen, kept_sigs, rows = set(), [], []
    for cat, doc_id, path in iter_docs(args.inp):
        paras = [p for p in read_doc(path).split("\n") if p.strip()]
        kept, dups = [], 0
        for p in paras:
            k = para_key(p)
            if k is None or k in seen:
                dups += 1
                continue
            seen.add(k)
            kept.append(p)
        sig = minhash(tokenize(" ".join(kept)))
        near, sim = "", 0.0
        if sig is not None:
            for other_id, other in kept_sigs:
                s = float(np.mean(sig == other))
                if s >= args.threshold and s > sim:
                    near, sim = other_id, s
        if near:
            kept = []
        elif sig is not None:
            kept_sigs.append((doc_id, sig))
        write_doc(f"{args.out}/{cat}/{doc_id}.txt", "\n".join(kept) + ("\n" if kept else ""))
        rows.append({"doc_id": doc_id, "category": cat, "paragraphs_in": len(paras),
                     "exact_duplicate_paragraphs": dups, "near_duplicate_of": near,
                     "jaccard_estimate": f"{sim:.3f}" if near else "", "paragraphs_out": len(kept)})

    with open(args.log, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    pin = sum(r["paragraphs_in"] for r in rows)
    dup = sum(r["exact_duplicate_paragraphs"] for r in rows)
    print(f"paragraphs in {pin:,}; exact duplicates removed {dup:,} ({100 * dup / pin:.1f}%); "
          f"near-duplicate documents {sum(bool(r['near_duplicate_of']) for r in rows)}; "
          f"paragraphs out {sum(r['paragraphs_out'] for r in rows):,}")
    for r in rows:
        if r["near_duplicate_of"]:
            print(f"  {r['doc_id']} ~ {r['near_duplicate_of']} (J ~ {r['jaccard_estimate']})")


if __name__ == "__main__":
    main()
