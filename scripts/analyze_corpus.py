"""Compute category/combined corpus statistics and reproducible Matplotlib figures.

Usage: python scripts/analyze_corpus.py [--stage extracted|cleaned|final]
Raw/cleaned iteration reports are separate from final results. Numbers are kept
by default; --numbers normalize replaces numeric tokens with <NUM> for counts.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from itertools import accumulate

import numpy as np
import pandas as pd
import regex

if __package__:
    from . import corpus_common as core
    from .segment_sentences import segment_sentences
else:
    import corpus_common as core
    from segment_sentences import segment_sentences

COVERAGE_TARGETS = (.10, .25, .50, .75, .90, .95, .99)


def frequency_rows(counts: Counter, scope: str) -> list[dict]:
    total, cumulative = sum(counts.values()), 0
    rows = []
    for rank, (word, frequency) in enumerate(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])), 1):
        cumulative += frequency
        rows.append({"scope": scope, "rank": rank, "word": word, "frequency": frequency,
                     "relative_frequency": frequency / total,
                     "cumulative_coverage": cumulative / total})
    return rows


def coverage_rows(counts: Counter, scope: str) -> list[dict]:
    ordered = sorted(counts.values(), reverse=True)
    cumulative = list(accumulate(ordered))
    total = sum(ordered)
    rows = []
    for target in COVERAGE_TARGETS:
        # Integer arithmetic avoids an exact threshold becoming 3.00000000004.
        needed = (total * round(target * 100) + 99) // 100
        types = int(np.searchsorted(cumulative, needed, side="left")) + 1 if total else 0
        rows.append({"scope": scope, "target_coverage": target, "types_required": types,
                     "achieved_coverage": cumulative[types - 1] / total if total else 0,
                     "total_tokens": total})
    return rows


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {key: 0 for key in ("sentences", "min", "max", "mean", "median", "std", "p10", "p25", "p50", "p75", "p90", "p95", "p99")}
    values = np.array(lengths)
    result = {"sentences": len(lengths), "min": int(values.min()), "max": int(values.max()),
              "mean": float(values.mean()), "median": float(np.median(values)),
              "std": float(values.std(ddof=0))}
    result.update({f"p{p}": float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90, 95, 99)})
    return result


def word_lengths(counts: Counter, scope: str) -> list[dict]:
    distributions = {"graphemes": {}, "codepoints": {}}
    for word, frequency in counts.items():
        for metric, size in (("graphemes", len(regex.findall(r"\X", word))), ("codepoints", len(word))):
            row = distributions[metric].setdefault(size, {"scope": scope, "unit": metric, "length": size,
                                                        "type_frequency": 0, "token_frequency": 0})
            row["type_frequency"] += 1
            row["token_frequency"] += frequency
    return [distributions[metric][length] for metric in distributions for length in sorted(distributions[metric])]


def dataframe_csv(path, rows, fields):
    frame = pd.DataFrame(rows, columns=fields)
    core.atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def plots(directory, scope, counts, sentence_lengths, lengths, growth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    values = sorted(counts.values(), reverse=True)
    ranks = np.arange(1, len(values) + 1)
    def save(name, xlabel, ylabel, draw):
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        draw(axis)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=f"{scope}: {name.replace('_', ' ')}")
        axis.grid(alpha=.2)
        figure.savefig(directory / f"{name}.png", dpi=150)
        plt.close(figure)
    def sentence_plot(ax):
        cutoff = float(np.percentile(sentence_lengths, 99)) if sentence_lengths else 0
        if len(sentence_lengths) > 100 and max(sentence_lengths) > max(3 * cutoff, 50):
            visible = [length for length in sentence_lengths if length <= cutoff]
            ax.hist(visible, bins=min(60, max(1, len(set(visible)))))
            ax.text(.03, .94, f"Main view: up to 99th percentile ({cutoff:g} tokens)\n"
                    f"{len(sentence_lengths) - len(visible)} longer segments shown in inset",
                    transform=ax.transAxes, va="top", fontsize=8)
            inset = ax.inset_axes([.55, .48, .4, .38])
            inset.hist(sentence_lengths, bins=60)
            inset.set_yscale("log")
            inset.set_title("All lengths (log count)", fontsize=8)
            inset.tick_params(labelsize=7)
        else:
            ax.hist(sentence_lengths, bins=min(60, max(1, len(set(sentence_lengths)))))
    save("sentence_length_distribution", "Word tokens per sentence", "Sentences", sentence_plot)
    graphemes = [r for r in lengths if r["unit"] == "graphemes"]
    save("word_length_distribution", "Unicode grapheme clusters per token", "Token frequency",
         lambda ax: ax.bar([r["length"] for r in graphemes], [r["token_frequency"] for r in graphemes]))
    spectrum = Counter(values)
    def frequency_plot(ax):
        if spectrum:
            ax.loglog(sorted(spectrum), [spectrum[k] for k in sorted(spectrum)], ".")
    save("word_frequency_distribution", "Occurrences of a word type (log)", "Number of types (log)", frequency_plot)
    save("rank_frequency", "Type rank (log)", "Frequency (log)",
         lambda ax: ax.loglog(ranks, values) if values else None)
    save("cumulative_coverage", "Highest-frequency types included", "Fraction of all tokens",
         lambda ax: ax.plot(ranks, np.cumsum(values) / sum(values)) if values else None)
    save("vocabulary_growth", "Tokens encountered (document order)", "Unique types seen",
         lambda ax: ax.plot([p[0] for p in growth], [p[1] for p in growth]) if growth else None)


def main():
    parser = core.parser(__doc__)
    parser.add_argument("--stage", choices=["extracted", "cleaned", "final"], default="final")
    parser.add_argument("--out", help="Report directory; default results/ or results/iterations/<stage>/")
    parser.add_argument("--numbers", choices=["keep", "normalize", "exclude"], default="keep")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = core.setup(args)
    source_stage = "segmented" if args.stage == "final" else args.stage
    documents, dependencies = core.load_stage(root, source_stage)
    if args.stage == "final":
        included = [d for d in documents if d["included"]]
        sentence_frame = pd.read_csv(root / "data/processed/combined/sentences.tsv", sep="\t", keep_default_na=False)
    else:
        included = [d for d in documents if d["processing_status"] in {"extracted", "cleaned"}]
        for document in included:
            if args.stage == "extracted":
                document["text"] = core.resolve(root, document["extracted_file"]).read_text(encoding="utf-8")
        sentence_frame = pd.DataFrame([
            {"document_id": d["document_id"], "category": d["category"], "sentence": sentence}
            for d in included for sentence in segment_sentences(d["text"])],
            columns=["document_id", "category", "sentence"])
    out = core.resolve(root, args.out or ("results" if args.stage == "final" else f"results/iterations/{args.stage}"))
    core.protect_raw(root, out)
    for document in documents:
        source_file = core.resolve(root, document["source_file"])
        if source_file.is_relative_to(out):
            raise ValueError("Report directory must not contain source documents")
    all_frequencies, all_coverage, all_lengths, all_sentence_stats, summaries = [], [], [], [], []
    all_docs = sorted(included, key=lambda d: (d["category"], d["document_id"]))
    for scope in ["combined", *core.CATEGORIES]:
        selected = all_docs if scope == "combined" else [d for d in all_docs if d["category"] == scope]
        counts, original_counts = Counter(), Counter()
        seen, growth, position = set(), [], 0
        for document in selected:
            original_words = core.tokens(document["text"])
            original_counts.update(original_words)
            words = core.tokens(document["text"], args.numbers)
            counts.update(words)
            for word in words:
                position += 1
                seen.add(word)
                if position % 1000 == 0:
                    growth.append((position, len(seen)))
        if position:
            growth.append((position, len(seen)))
        sentences = sentence_frame if scope == "combined" else sentence_frame[sentence_frame["category"] == scope]
        sentence_lengths = [len(core.tokens(text, args.numbers)) for text in sentences["sentence"]]
        stats = length_stats(sentence_lengths)
        total = sum(counts.values())
        summaries.append({"scope": scope, "documents": len(selected), "tokens": total,
                          "types": len(counts), "type_token_ratio": len(counts) / total if total else 0,
                          "hapax_legomena": sum(v == 1 for v in counts.values()),
                          "mean_sentence_length": stats["mean"], "sentences": stats["sentences"],
                          "provisional_documents": sum(d.get("provisional", False) for d in selected)
                              if args.stage == "final" else len(selected)})
        all_frequencies.extend(frequency_rows(counts, scope))
        all_coverage.extend(coverage_rows(counts, scope))
        # Physical word lengths always use original forms, never the literal <NUM> placeholder.
        physical_counts = Counter({word: frequency for word, frequency in original_counts.items()
                                   if args.numbers != "exclude" or not word[0].isnumeric()})
        lengths = word_lengths(physical_counts, scope)
        all_lengths.extend(lengths)
        all_sentence_stats.append({"scope": scope, **stats})
        if not args.no_plots:
            plots(out / "plots" / scope, scope, counts, sentence_lengths, lengths, growth)
    total = summaries[0]["tokens"]
    for summary in summaries:
        summary["token_contribution"] = summary["tokens"] / total if total else 0
    dataframe_csv(out / "word_frequencies.csv", all_frequencies,
                  ["scope", "rank", "word", "frequency", "relative_frequency", "cumulative_coverage"])
    dataframe_csv(out / "coverage_analysis.csv", all_coverage,
                  ["scope", "target_coverage", "types_required", "achieved_coverage", "total_tokens"])
    dataframe_csv(out / "word_length_distribution.csv", all_lengths,
                  ["scope", "unit", "length", "type_frequency", "token_frequency"])
    dataframe_csv(out / "sentence_length_statistics.csv", all_sentence_stats,
                  ["scope", "sentences", "min", "max", "mean", "median", "std", "p10", "p25", "p50", "p75", "p90", "p95", "p99"])
    dataframe_csv(out / "category_comparison.csv", summaries, list(summaries[0]))
    dominant = [s["scope"] for s in summaries[1:] if s["token_contribution"] > .6]
    note = f"Category dominance (>60% of tokens): {', '.join(dominant)}" if dominant else "No category exceeds 60% of tokens."
    provisional = args.stage != "final" or any(d.get("provisional") for d in included)
    lines = ["Devanagari Konkani corpus statistics", f"Input stage: {args.stage}",
             f"Provisional (contains unreviewed text): {provisional}", f"Number policy: {args.numbers}",
             "Tokens: Unicode letter/mark words (internal hyphens/apostrophes retained) and numeric runs.",
             "No case folding, stemming, stopword removal or spelling correction.",
             "Sentence standard deviation: population (ddof=0); percentiles: linear interpolation.",
             "Word length: regex \\X extended grapheme clusters; codepoints also reported.",
             "Grapheme clusters are not a validated linguistic akshara segmentation.",
             "Word-length tables use original numeric forms even with --numbers normalize.",
             "Empty scopes use zero counts/statistics; coverage targets are not achieved with zero tokens.",
             note, "", pd.DataFrame(summaries).to_string(index=False), "", "Most frequent words (combined):"]
    lines.extend(f"{r['rank']}. {r['word']}: {r['frequency']}" for r in all_frequencies if r["scope"] == "combined" and r["rank"] <= 20)
    core.atomic_text(out / "corpus_statistics.txt", "\n".join(lines) + "\n")
    core.atomic_text(out / "analysis_manifest.json", json.dumps({"stage": args.stage, "numbers": args.numbers,
                      "provisional": provisional, "dependencies": dependencies, "plots_generated": not args.no_plots,
                      "analysis_script_sha256": core.sha256_file(core.ROOT / "scripts/analyze_corpus.py"),
                      "regex_version": regex.__version__, "pandas_version": pd.__version__,
                      "numpy_version": np.__version__}, indent=2, sort_keys=True) + "\n")
    if not args.no_plots:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        axis.bar([int(s["scope"].split("_", 1)[0]) for s in summaries[1:]], [s["tokens"] for s in summaries[1:]])
        axis.set_xticks(range(1, 8))
        axis.set(xlabel="Category number (see category_comparison.csv)", ylabel="Word tokens", title="Category token distribution")
        figure.savefig(out / "plots/category_token_distribution.png", dpi=150)
        plt.close(figure)
    logging.info("Analysed %d documents, %d tokens, %d types; %s", len(included), total, summaries[0]["types"], note)


if __name__ == "__main__":
    core.cli(main)
