"""Type-token, frequency and coverage analysis of one corpus iteration.

Input is a document tree (<root>/<category>/<doc_id>.txt). Tokens are words
with punctuation removed and every number mapped to <NUM> (corpus_utils.tokenize).

Outputs in --out:
    corpus_statistics.txt   human-readable summary, overall and per category
    stats.json              same numbers, machine-readable (used by the cleaning log and report)
    word_frequencies.csv    rank, type, freq, rel_freq, cum_coverage (decreasing frequency)
    coverage_analysis.csv   types needed for 10..99% token coverage, overall and per category
    category_statistics.csv one row per category
    plots/{zipf,heaps,coverage}.png

Usage:
    python scripts/analyze_corpus.py --in data/interim/iter0 --out results/iter0 --label "iteration 0 (raw)"
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from corpus_utils import iter_docs, read_doc, tokenize

COVERAGE_LEVELS = [10, 25, 50, 75, 90, 95, 99]
KINDS = ["deva", "num", "latin", "other"]


def basic_stats(counts, kinds):
    n = sum(counts.values())
    v = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    dis = sum(1 for c in counts.values() if c == 2)
    return {
        "tokens": n, "types": v,
        "ttr": v / n if n else 0.0,
        "root_ttr": v / math.sqrt(n) if n else 0.0,
        "hapax": hapax, "hapax_ratio_types": hapax / v if v else 0.0,
        "hapax_ratio_tokens": hapax / n if n else 0.0,
        "dis_legomena": dis,
        "token_share": {k: kinds[k] / n if n else 0.0 for k in KINDS},
    }


def coverage(counts):
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.int64)
    cum = np.cumsum(freqs) / freqs.sum()
    out = {}
    for lvl in COVERAGE_LEVELS:
        k = int(np.searchsorted(cum, lvl / 100 - 1e-12) + 1)
        out[lvl] = k
    return out, cum


def heaps_curve(token_stream, points=200):
    seen, xs, ys = set(), [], []
    n = len(token_stream)
    marks = set(np.unique(np.geomspace(1, n, points).astype(int)))
    for i, t in enumerate(token_stream, 1):
        seen.add(t)
        if i in marks:
            xs.append(i)
            ys.append(len(seen))
    return np.array(xs), np.array(ys)


def fit_loglog(x, y):
    """Least-squares fit of log y = a + b log x; returns (exp(a), b)."""
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    return float(math.exp(a)), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    overall, kinds_all, stream = Counter(), Counter(), []
    per_cat, kinds_cat, docs_cat = {}, {}, Counter()
    for cat, _, path in iter_docs(args.inp):
        toks = tokenize(read_doc(path), keep_kind=True)
        words = [t for t, _ in toks]
        kc = Counter(k for _, k in toks)
        overall.update(words)
        kinds_all.update(kc)
        stream.extend(words)
        per_cat.setdefault(cat, Counter()).update(words)
        kinds_cat.setdefault(cat, Counter()).update(kc)
        docs_cat[cat] += 1
    if not overall:
        raise SystemExit(f"no tokens found under {args.inp}")

    stats = {"label": args.label, "input": args.inp, "documents": sum(docs_cat.values()),
             "overall": basic_stats(overall, kinds_all), "categories": {}}
    cov_all, cum_all = coverage(overall)
    stats["overall"]["coverage"] = cov_all
    for cat, c in per_cat.items():
        s = basic_stats(c, kinds_cat[cat])
        s["documents"] = docs_cat[cat]
        s["share_of_corpus"] = s["tokens"] / stats["overall"]["tokens"]
        s["coverage"] = coverage(c)[0]
        stats["categories"][cat] = s

    # Zipf (fit over ranks 10..10000 where the law usually holds) and Heaps
    freqs = np.array(sorted(overall.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    lo, hi = 10, min(10000, len(freqs))
    zipf_c, zipf_slope = fit_loglog(ranks[lo - 1:hi], freqs[lo - 1:hi]) if hi > lo else (0.0, 0.0)
    hx, hy = heaps_curve(stream)
    sel = hx >= 1000
    heaps_k, heaps_beta = fit_loglog(hx[sel], hy[sel]) if sel.sum() > 2 else (0.0, 0.0)
    stats["zipf"] = {"slope": zipf_slope, "fit_ranks": [lo, hi]}
    stats["heaps"] = {"K": heaps_k, "beta": heaps_beta}

    # ---- files
    total = stats["overall"]["tokens"]
    with open(out / "word_frequencies.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rank", "type", "freq", "rel_freq", "cum_coverage"])
        cum = 0
        for r, (t, c) in enumerate(sorted(overall.items(), key=lambda x: (-x[1], x[0])), 1):
            cum += c
            w.writerow([r, t, c, f"{c / total:.8f}", f"{cum / total:.6f}"])

    with open(out / "coverage_analysis.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["scope", "coverage_pct", "types_needed", "pct_of_vocabulary"])
        for scope, s in [("overall", stats["overall"])] + sorted(stats["categories"].items()):
            for lvl, k in s["coverage"].items():
                w.writerow([scope, lvl, k, f"{100 * k / s['types']:.3f}"])

    cat_fields = ["category", "documents", "tokens", "share_of_corpus", "types", "ttr", "root_ttr",
                  "hapax", "hapax_ratio_types", "deva_share", "num_share", "latin_share", "other_share"]
    with open(out / "category_statistics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cat_fields)
        for cat, s in sorted(stats["categories"].items()):
            w.writerow([cat, s["documents"], s["tokens"], f"{s['share_of_corpus']:.4f}", s["types"],
                        f"{s['ttr']:.4f}", f"{s['root_ttr']:.2f}", s["hapax"], f"{s['hapax_ratio_types']:.4f}"]
                       + [f"{s['token_share'][k]:.4f}" for k in KINDS])

    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    o = stats["overall"]
    top = sorted(overall.items(), key=lambda x: -x[1])[:30]
    lines = [
        f"Corpus statistics: {args.label}".rstrip(": "),
        f"Input: {args.inp}",
        "",
        f"Documents:            {stats['documents']:,}",
        f"Tokens (N):           {o['tokens']:,}",
        f"Types (V):            {o['types']:,}",
        f"Type-token ratio:     {o['ttr']:.4f}",
        f"Root TTR (V/sqrt N):  {o['root_ttr']:.2f}",
        f"Hapax legomena:       {o['hapax']:,} ({100 * o['hapax_ratio_types']:.1f}% of types, "
        f"{100 * o['hapax_ratio_tokens']:.2f}% of tokens)",
        f"Dis legomena:         {o['dis_legomena']:,}",
        "Token composition:    " + ", ".join(f"{k} {100 * o['token_share'][k]:.2f}%" for k in KINDS),
        f"Zipf slope (ranks {lo}-{hi}): {zipf_slope:.3f}",
        f"Heaps' law V = K N^b: K = {heaps_k:.2f}, b = {heaps_beta:.3f}",
        "",
        "Coverage (types needed for X% of tokens):",
    ] + [f"  {lvl:>3d}%: {k:>8,d} types ({100 * k / o['types']:.2f}% of vocabulary)"
         for lvl, k in cov_all.items()] + [
        "",
        f"{'category':24s} {'docs':>5s} {'tokens':>11s} {'share':>6s} {'types':>9s} {'TTR':>7s} {'hapax%':>7s} {'deva%':>6s}",
    ] + [f"{c:24s} {s['documents']:5d} {s['tokens']:11,d} {100 * s['share_of_corpus']:5.1f}% {s['types']:9,d} "
         f"{s['ttr']:7.4f} {100 * s['hapax_ratio_types']:6.1f}% {100 * s['token_share']['deva']:5.1f}%"
         for c, s in sorted(stats["categories"].items())] + [
        "",
        "Top 30 types:",
    ] + [f"  {i:>3d}. {t}\t{c:,}" for i, (t, c) in enumerate(top, 1)]
    (out / "corpus_statistics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:22]))

    if not args.no_plots:
        plot(out / "plots", ranks, freqs, cum_all, hx, hy, heaps_k, heaps_beta, zipf_slope, args.label)


def plot(pdir, ranks, freqs, cum, hx, hy, k, beta, slope, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir.mkdir(parents=True, exist_ok=True)
    suffix = f" ({label})" if label else ""

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(ranks, freqs, ".", ms=2, alpha=0.5, label="observed")
    ax.loglog(ranks, freqs[0] * ranks ** -1.0, "--", lw=1, label="ideal Zipf (slope -1)")
    ax.set(xlabel="rank (log)", ylabel="frequency (log)", title=f"Zipf rank-frequency{suffix}")
    ax.text(0.03, 0.04, f"fitted slope = {slope:.2f}", transform=ax.transAxes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "zipf.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(hx, hy, label="observed vocabulary")
    ax.plot(hx, k * hx ** beta, "--", lw=1, label=f"Heaps fit V = {k:.1f} N^{beta:.2f}")
    ax.set(xlabel="tokens read (N)", ylabel="distinct types (V)", title=f"Vocabulary growth{suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "heaps.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogx(np.arange(1, len(cum) + 1), 100 * cum)
    for lvl in COVERAGE_LEVELS:
        ax.axhline(lvl, color="grey", lw=0.4)
    ax.set(xlabel="top-N types (log)", ylabel="% of tokens covered", title=f"Token coverage{suffix}")
    fig.tight_layout()
    fig.savefig(pdir / "coverage.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
