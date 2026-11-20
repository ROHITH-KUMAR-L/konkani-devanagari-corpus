"""Sentence-length and word-length distributions of the processed corpus (PRD FR-L1..L3).

Sentence length is counted in word tokens (corpus_utils.tokenize). Word length
is measured for Devanagari word tokens (numbers excluded) in
  - characters: Unicode code points (a matra or virama counts as one), and
  - aksharas: orthographic syllables (corpus_utils.aksharas),
both token-weighted (every occurrence) and type-weighted (each distinct word once).

Outputs in --out:
    length_analysis.txt              summary
    sentence_length_stats.csv        overall and per category
    sentence_length_distribution.csv length -> count
    word_length_distribution.csv     length -> counts for chars/aksharas x token/type
    plots/{sentence_length,word_length}.png

Usage:
    python scripts/length_analysis.py --processed data/processed --out results
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from corpus_utils import CATEGORIES, NUM, aksharas, tokenize

PCTS = [5, 25, 50, 75, 95, 99]


def describe(lengths):
    a = np.asarray(lengths)
    d = {"sentences": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
         "min": int(a.min()), "max": int(a.max())}
    for p in PCTS:
        d[f"p{p}"] = float(np.percentile(a, p))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    sent_len = {c: [] for c in CATEGORIES}
    tok_chars, tok_aks, types = Counter(), Counter(), Counter()
    for cat in CATEGORIES:
        tsv = Path(args.processed) / "by_category" / cat / "sentences.tsv"
        if not tsv.exists():
            continue
        with open(tsv, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                sent_len[cat].append(int(row["n_tokens"]))
                for t in tokenize(row["sentence"]):
                    if t != NUM:
                        types[t] += 1
    for t, c in types.items():
        tok_chars[len(t)] += c
        tok_aks[len(aksharas(t))] += c
    type_chars = Counter(len(t) for t in types)
    type_aks = Counter(len(aksharas(t)) for t in types)

    sent_len = {c: v for c, v in sent_len.items() if v}
    all_len = [x for v in sent_len.values() for x in v]
    stats = {"overall": describe(all_len)} | {c: describe(v) for c, v in sent_len.items()}

    with open(out / "sentence_length_stats.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        keys = list(stats["overall"])
        w.writerow(["scope"] + keys)
        for scope, d in stats.items():
            w.writerow([scope] + [f"{d[k]:.2f}" if isinstance(d[k], float) else d[k] for k in keys])
    dist = Counter(all_len)
    with open(out / "sentence_length_distribution.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n_tokens", "sentences"])
        w.writerows(sorted(dist.items()))
    with open(out / "word_length_distribution.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["length", "chars_token_weighted", "chars_type_weighted",
                    "aksharas_token_weighted", "aksharas_type_weighted"])
        for n in range(1, max(max(tok_chars), max(tok_aks)) + 1):
            w.writerow([n, tok_chars[n], type_chars[n], tok_aks[n], type_aks[n]])

    def mean(c):
        return sum(k * v for k, v in c.items()) / sum(c.values())

    def median(c):
        half, run = sum(c.values()) / 2, 0
        for k in sorted(c):
            run += c[k]
            if run >= half:
                return k

    o = stats["overall"]
    lines = [
        "Length analyses (processed corpus)", "",
        "Sentence length in word tokens:",
        f"  sentences {o['sentences']:,}; mean {o['mean']:.2f}; std {o['std']:.2f}; median {o['p50']:.0f}; "
        f"p5 {o['p5']:.0f}; p95 {o['p95']:.0f}; p99 {o['p99']:.0f}; max {o['max']}",
        f"  sentences with 1-2 tokens: {sum(1 for x in all_len if x <= 2):,} "
        f"({100 * sum(1 for x in all_len if x <= 2) / len(all_len):.1f}%); "
        f"> 60 tokens: {sum(1 for x in all_len if x > 60):,}",
    ] + [f"  {c:24s} n {d['sentences']:>8,d}  mean {d['mean']:6.2f}  median {d['p50']:4.0f}  "
         f"std {d['std']:6.2f}  p95 {d['p95']:4.0f}" for c, d in stats.items() if c != "overall"] + [
        "",
        "Word length (Devanagari words, numbers excluded):",
        f"  characters  token-weighted mean {mean(tok_chars):.2f} median {median(tok_chars)}; "
        f"type-weighted mean {mean(type_chars):.2f} median {median(type_chars)}",
        f"  aksharas    token-weighted mean {mean(tok_aks):.2f} median {median(tok_aks)}; "
        f"type-weighted mean {mean(type_aks):.2f} median {median(type_aks)}",
        f"  characters per akshara (token-weighted): {mean(tok_chars) / mean(tok_aks):.2f}",
    ]
    (out / "length_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(1, 82)
    for c, v in sent_len.items():
        ax.hist(np.clip(v, 1, 80), bins=bins, histtype="step", density=True, label=c)
    ax.set(xlabel="sentence length (tokens; 80 = 80 or more)", ylabel="share of sentences",
           title="Sentence-length distribution by category")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "sentence_length.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (tok, typ, unit, lim) in zip(axes, [(tok_chars, type_chars, "characters (code points)", 25),
                                                 (tok_aks, type_aks, "aksharas", 12)]):
        xs = np.arange(1, lim + 1)
        tt, ty = sum(tok.values()), sum(typ.values())
        ax.bar(xs - 0.2, [tok[x] / tt for x in xs], 0.4, label="token-weighted")
        ax.bar(xs + 0.2, [typ[x] / ty for x in xs], 0.4, label="type-weighted")
        ax.set(xlabel=f"word length in {unit}", ylabel="share", title=f"Word length: {unit}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "word_length.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
