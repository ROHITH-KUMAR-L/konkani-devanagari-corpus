"""Conservatively clean the extraction manifest; retain minimal and analysis versions.

Usage: python scripts/clean_text.py [--drop-page-numbers] [--min-line-deva 0.7]
Potentially destructive rules are opt-in and every removed line is logged.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import regex

if __package__:
    from . import corpus_common as core
    from .extract_text import html_text
else:
    import corpus_common as core
    from extract_text import html_text


def clean_document(text: str, *, drop_page_numbers=False, repeated_headers=False,
                   min_line_deva=0.0, rejoin_lines=False, dehyphenate=False,
                   remove_patterns=()) -> tuple[str, str, list[dict]]:
    text = core.normalize(text)
    # Preserve combining marks, ZWJ/ZWNJ and form-feed page boundaries.
    text = regex.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", "", text)
    text = text.replace("\u00ad", "").replace("\u200b", "")
    if regex.search(r"</?(?:html|body|p|div|pre|br|span|script|style|nav)\b", text, regex.I):
        text = html_text(text)
    text = regex.sub(r"[^\S\n\f]+", " ", text)
    minimal = "\n".join(line.strip(" \t") for line in text.split("\n")).strip(" \t\n")
    pages = minimal.split("\f")
    repeated = set()
    if repeated_headers and len(pages) >= 3:
        edges = Counter()
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            edges.update(set(lines[:2] + lines[-2:]))
        repeated = {line for line, count in edges.items() if count >= max(3, len(pages) * .6)}
    compiled = [regex.compile(pattern) for pattern in remove_patterns]
    removed, cleaned_pages = [], []
    for page_no, page in enumerate(pages, 1):
        lines = page.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edge_positions = set(nonempty[:2] + nonempty[-2:])
        kept = []
        for line_no, line in enumerate(lines, 1):
            reason = ""
            if line.strip():
                if drop_page_numbers and regex.fullmatch(r"[-–— ]*\p{Nd}{1,6}[-–— ]*", line):
                    reason = "standalone_page_number_candidate"
                elif line in repeated and line_no - 1 in edge_positions:
                    reason = "repeated_page_edge"
                elif any(pattern.search(line) for pattern in compiled):
                    reason = "explicit_removal_pattern"
                elif min_line_deva and core.script_stats(line)["letters"] >= 8 and core.script_stats(line)["devanagari_ratio"] < min_line_deva:
                    reason = "low_devanagari_line"
            if reason:
                removed.append({"page": page_no, "line": line_no, "reason": reason, "text": line})
            else:
                kept.append(line)
        cleaned_pages.append("\n".join(kept))
    analysis = "\n\n".join(cleaned_pages)
    if dehyphenate:
        analysis = regex.sub(r"(?<=[\p{L}\p{M}])-\n(?=\p{L})", "", analysis)
    if rejoin_lines:
        analysis = regex.sub(r"(?<!\n)\n(?!\n)", " ", analysis)
    analysis = regex.sub(r"[^\S\n]+([.,?!;:।॥])", r"\1", analysis)
    analysis = regex.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", analysis)
    return minimal, analysis.strip(), removed


def main():
    parser = core.parser(__doc__)
    parser.add_argument("--drop-page-numbers", action="store_true")
    parser.add_argument("--repeated-headers", action="store_true", help="Requires form-feed page boundaries")
    parser.add_argument("--min-line-deva", type=float, default=0)
    parser.add_argument("--rejoin-lines", action="store_true", help="Join single newlines; use cautiously for verse/transcripts")
    parser.add_argument("--dehyphenate", action="store_true")
    parser.add_argument("--remove-pattern", action="append", default=[], help="Regex for manually identified boilerplate; repeatable")
    args = parser.parse_args()
    root = core.setup(args)
    if not 0 <= args.min_line_deva <= 1:
        parser.error("--min-line-deva must be in [0,1]")
    documents, dependencies = core.load_stage(root, "extracted")
    options = {"drop_page_numbers": args.drop_page_numbers, "repeated_headers": args.repeated_headers,
               "min_line_deva": args.min_line_deva, "rejoin_lines": args.rejoin_lines,
               "dehyphenate": args.dehyphenate, "remove_patterns": args.remove_pattern}
    changes, log_rows = [], []
    for document in documents:
        if document["processing_status"] != "extracted":
            continue
        text = core.resolve(root, document["extracted_file"]).read_text(encoding="utf-8")
        minimal, analysis, removed = clean_document(text, **options)
        base = root / "data/processed/documents" / document["category"]
        minimal_path = base / f"{document['document_id']}.minimal.txt"
        clean_path = base / f"{document['document_id']}.clean.txt"
        core.atomic_text(minimal_path, minimal + "\n")
        core.atomic_text(clean_path, analysis + "\n" if analysis else "")
        document.update(text=analysis, minimal_file=core.relative(root, minimal_path),
                        clean_file=core.relative(root, clean_path), processing_status="cleaned",
                        clean_sha256=core.text_hash(analysis), raw_tokens=len(core.tokens(text)),
                        cleaned_tokens=len(core.tokens(analysis)), removed_lines=len(removed))
        changes.extend({"document_id": document["document_id"], **row} for row in removed)
        log_rows.append(f"| {document['document_id']} | {document['raw_tokens']} | {document['cleaned_tokens']} | {len(removed)} |")
    core.atomic_text(root / "results/cleaning_changes.jsonl", core.jsonl_text(changes))
    core.save_stage(root, "cleaned", documents, dependencies, {"script": "clean_text", **options})
    log = root / "docs/data_cleaning_log.md"
    previous = log.read_text(encoding="utf-8") if log.exists() else "# Data cleaning log\n"
    fingerprint = core.text_hash(core.jsonl_text(documents) + json.dumps(options, sort_keys=True))[:16]
    marker = f"<!-- pipeline-cleaning:{fingerprint} -->"
    if marker not in previous:
        block = (f"\n\n{marker}\n## Pipeline cleaning run {fingerprint}\n\n"
                 f"Options: `{json.dumps(options, ensure_ascii=False, sort_keys=True)}`\n\n"
                 "| Document | Input tokens | Cleaned tokens | Removed lines |\n"
                 "|---|---:|---:|---:|\n" + "\n".join(log_rows) + "\n\n"
                 "Removed text: `results/cleaning_changes.jsonl`. Minimal text is retained per document.\n")
        core.atomic_text(log, previous + block)
    logging.info("Cleaned %d documents; logged %d removed lines", len(log_rows), len(changes))


if __name__ == "__main__":
    core.cli(main)
