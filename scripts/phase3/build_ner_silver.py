"""Turn the silver span annotations into CoNLL and split them into dev/test.

Input:  data/eval/ner/pool.conll          (from sample_ner_sentences.py, all tags O)
        data/eval/ner/silver_spans.txt    (sent_id<TAB>TYPE:tok tok | TYPE:tok ...)
Output: data/eval/ner/silver_all.conll, silver_dev.conll, silver_test.conll
        results/phase3/ner_label_counts.csv

Each span is matched to the first occurrence of its token sequence whose tokens
are still untagged, so a name repeated in a sentence is listed once per
mention. Any span that cannot be matched is a hard error.

The split is 50/50 within each category (seed 2026), so dev and test have the
same domain mix. THE SILVER LABELS ARE AN AI FIRST PASS AND MUST BE VERIFIED.
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_common import LABELS, Sentence, read_conll, spans, write_conll  # noqa: E402

HEADER = [
    "SILVER NER DATA: AI first-pass annotation (Claude), NOT gold.",
    "Every sentence must be verified by a team annotator before use as gold.",
    "Guidelines: docs/ner_guidelines.md. Tags: BIO over PER LOC ORG MISC DATE.",
]


def parse_spans(path):
    ann = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sid, _, rest = line.partition("\t")
        items = []
        for part in filter(None, (p.strip() for p in rest.split("|"))):
            typ, _, surf = part.partition(":")
            assert typ in LABELS, (sid, part)
            items.append((typ, surf.split()))
        ann[sid] = items
    return ann


def apply(sent, items):
    tags = ["O"] * len(sent.tokens)
    for typ, seq in items:
        n = len(seq)
        for i in range(len(sent.tokens) - n + 1):
            if sent.tokens[i:i + n] == seq and all(t == "O" for t in tags[i:i + n]):
                tags[i] = "B-" + typ
                for j in range(i + 1, i + n):
                    tags[j] = "I-" + typ
                break
        else:
            raise SystemExit(f"{sent.meta['sent_id']}: span not found: {typ}:{' '.join(seq)}")
    return Sentence(dict(sent.meta, annotation="silver-ai-first-pass"), sent.tokens, tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/eval/ner/pool.conll")
    ap.add_argument("--spans", default="data/eval/ner/silver_spans.txt")
    ap.add_argument("--outdir", default="data/eval/ner")
    ap.add_argument("--counts", default="results/phase3/ner_label_counts.csv")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    ann = parse_spans(a.spans)
    pool = {s.meta["sent_id"]: s for s in read_conll(a.pool)}
    silver = [apply(pool[sid], items) for sid, items in ann.items()]

    rng = random.Random(a.seed)
    by_cat = defaultdict(list)
    for s in silver:
        by_cat[s.meta["category"]].append(s)
    dev, test = [], []
    for c in sorted(by_cat):
        items = by_cat[c][:]
        rng.shuffle(items)
        h = (len(items) + 1) // 2
        dev += items[:h]
        test += items[h:]
    key = lambda s: s.meta["sent_id"]  # noqa: E731
    dev.sort(key=key)
    test.sort(key=key)

    out = Path(a.outdir)
    write_conll(out / "silver_all.conll", silver, HEADER)
    write_conll(out / "silver_dev.conll", dev, HEADER)
    write_conll(out / "silver_test.conll", test, HEADER)

    Path(a.counts).parent.mkdir(parents=True, exist_ok=True)
    with open(a.counts, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "sentences", "tokens", "sentences_with_entity"] + LABELS + ["total_entities"])
        for name, ss in [("all", silver), ("dev", dev), ("test", test)]:
            c = Counter(t for s in ss for t, _, _ in spans(s.tags))
            w.writerow([name, len(ss), sum(len(s.tokens) for s in ss),
                        sum(1 for s in ss if spans(s.tags))] + [c[l] for l in LABELS] + [sum(c.values())])
    print(Path(a.counts).read_text())


if __name__ == "__main__":
    main()
