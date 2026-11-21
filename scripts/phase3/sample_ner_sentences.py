"""Draw a stratified sample of corpus sentences for NER annotation (Phase 3).

Reads the combined sentence file and the document metadata, filters out
sentences that are too short, too long or not Devanagari, and samples from
every category. The output is CoNLL (token<TAB>tag, blank line between
sentences) with every tag set to "O", ready to import into an annotation tool
or to label by hand.

Allocation: each category gets a share proportional to sqrt(#candidate
sentences), with a floor (--min-per-cat). This keeps news (the most
entity-dense and by far the largest category) the biggest share without
drowning out literature, culture and web text.

Ordering: the sampled sentences are interleaved so that *any prefix* of the
output is itself stratified in the same proportions. The first 300 sentences
are therefore a fair stratified subset (used for the silver first pass), and
the team can annotate the rest in order.

Tokenisation follows scripts/corpus_utils.TOKEN_RE (punctuation dropped), but
numbers keep their surface form instead of being mapped to <NUM>, because
DATE entities need the digits.

Usage:
    python scripts/phase3/sample_ner_sentences.py \
        --sentences /abs/path/data/processed/combined/sentences.tsv \
        --documents /abs/path/data/processed/combined/documents.jsonl \
        --n 3000 --out data/eval/ner/pool.conll
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_utils import TOKEN_RE, devanagari_ratio  # noqa: E402

SEED = 2026


def ner_tokens(text):
    """Corpus tokenisation with number surface kept (see module docstring)."""
    return [m.group() for m in TOKEN_RE.finditer(text)]


def load_categories(documents_path):
    cats = {}
    with open(documents_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cats[d["doc_id"]] = d["category"]
    return cats


def candidates(sentences_path, cats, min_tok, max_tok, min_deva):
    by_cat = defaultdict(list)
    csv.field_size_limit(10**8)
    with open(sentences_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        assert header[:4] == ["doc_id", "sent_id", "n_tokens", "sentence"], header
        for row in reader:
            if len(row) < 4:
                continue
            doc_id, sent_id, _, sent = row[:4]
            toks = ner_tokens(sent)
            if not (min_tok <= len(toks) <= max_tok):
                continue
            if devanagari_ratio(sent)[0] < min_deva:
                continue
            by_cat[cats.get(doc_id, "unknown")].append((doc_id, sent_id, sent, toks))
    return by_cat


def allocate(sizes, n, floor):
    w = {c: math.sqrt(s) for c, s in sizes.items()}
    tot = sum(w.values())
    alloc = {c: min(sizes[c], max(floor, round(n * w[c] / tot))) for c in sizes}
    # trim/fill the largest categories so the total is exactly n (if possible)
    order = sorted(sizes, key=lambda c: -sizes[c])
    i = 0
    while sum(alloc.values()) != n and i < 10 * n:
        c = order[i % len(order)]
        if sum(alloc.values()) > n and alloc[c] > floor:
            alloc[c] -= 1
        elif sum(alloc.values()) < n and alloc[c] < sizes[c]:
            alloc[c] += 1
        i += 1
    return alloc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--out", default="data/eval/ner/pool.conll")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--min-per-cat", type=int, default=150)
    ap.add_argument("--min-tokens", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--min-deva", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cats = load_categories(a.documents)
    by_cat = candidates(a.sentences, cats, a.min_tokens, a.max_tokens, a.min_deva)
    sizes = {c: len(v) for c, v in sorted(by_cat.items())}
    alloc = allocate(sizes, a.n, a.min_per_cat)

    ranked = []
    for c in sorted(by_cat):
        pool = by_cat[c]
        pick = rng.sample(pool, alloc[c])
        for i, item in enumerate(pick):
            ranked.append(((i + rng.random()) / alloc[c], c, item))
    ranked.sort(key=lambda r: r[0])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for k, (_, c, (doc_id, sent_id, sent, toks)) in enumerate(ranked, 1):
            f.write(f"# sent_id = ner{k:05d}\n# source = {doc_id}:{sent_id}\n# category = {c}\n# text = {sent}\n")
            for t in toks:
                f.write(f"{t}\tO\n")
            f.write("\n")

    print(f"wrote {len(ranked)} sentences to {out}")
    print(f"{'category':24s} {'candidates':>10s} {'sampled':>8s}")
    for c in sizes:
        print(f"{c:24s} {sizes[c]:10d} {alloc[c]:8d}")


if __name__ == "__main__":
    main()
