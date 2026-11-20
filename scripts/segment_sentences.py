"""Segment reviewed/deduplicated documents and publish all category/combined exports.

Usage: python scripts/segment_sentences.py [--abbreviation 'श्री.']
Heuristic boundaries need manual evaluation on representative Konkani samples.
"""
from __future__ import annotations

import logging

import regex

if __package__:
    from . import corpus_common as core
else:
    import corpus_common as core

ABBREVIATIONS = {"डॉ.", "श्री.", "सौ.", "प्रा.", "क्र.", "पृ.", "उदा.", "इ.",
                 "आर्.पी.आय.",
                 "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "etc.", "e.g.", "i.e."}
SENTENCE_FIELDS = ["document_id", "sentence_id", "category", "sentence", "n_tokens"]


def protected_positions(text: str, abbreviations) -> set[int]:
    protected = set()
    patterns = [
        r"(?:https?://|www\.)\S+", r"[\w.+-]+@[\w.-]+\.[\p{L}]{2,}",
        r"(?<!\p{L})(?:[A-Z]\.\s*)+(?=\p{L})",
        r"(?<![\p{L}\p{M}])(?:[\p{Script=Devanagari}&&\p{L}]\p{M}*\.){2,}",
        r"(?<![\p{L}\p{M}])(?:" + "|".join(regex.escape(a) for a in sorted(abbreviations, key=len, reverse=True)) + r")",
    ]
    for pattern in patterns:
        for match in regex.finditer(pattern, text, regex.V1):
            end = match.end()
            if pattern.startswith(r"(?:https?") or "@" in pattern:
                end = match.start() + len(match.group().rstrip(".,!?।॥\"'”’)]}"))
            protected.update(range(match.start(), end))
    for match in regex.finditer(r"(?<=\p{N})\.(?=\p{N})", text):
        protected.add(match.start())
    return protected


def segment_sentences(text: str, abbreviations=None) -> list[str]:
    abbreviations = ABBREVIATIONS | set(abbreviations or [])
    sentences = []
    # A blank line is an explicit paragraph boundary. Single OCR line wraps are not.
    for paragraph in regex.split(r"\n\s*\n|\f", core.normalize(text)):
        paragraph = regex.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        protected = protected_positions(paragraph, abbreviations)
        start, position = 0, 0
        while position < len(paragraph):
            char = paragraph[position]
            if char not in ".?!।॥" or position in protected:
                position += 1
                continue
            end = position + 1
            while end < len(paragraph) and paragraph[end] in ".?!।॥":
                end += 1
            # Ellipses represent hesitation; keep them inside the current sentence.
            if set(paragraph[position:end]) == {"."} and end - position >= 2:
                position = end
                continue
            if char == "॥":
                verse = regex.match(r"\s*\p{N}+\s*॥", paragraph[end:])
                if verse:
                    end += verse.end()
            while end < len(paragraph) and paragraph[end] in '\"\'”’»)]}':
                end += 1
            sentence = paragraph[start:end].strip()
            if core.tokens(sentence):
                sentences.append(sentence)
            start = position = end
        remainder = paragraph[start:].strip()
        if core.tokens(remainder):
            sentences.append(remainder)
    return sentences


def sentence_rows(documents, abbreviations=None):
    for document in documents:
        for number, sentence in enumerate(segment_sentences(document["text"], abbreviations), 1):
            yield {"document_id": document["document_id"],
                   "sentence_id": f"{document['document_id']}_S{number:06d}",
                   "category": document["category"], "sentence": sentence,
                   "n_tokens": len(core.tokens(sentence))}


def segmentation_flags(sentence: str) -> list[str]:
    flags = []
    if len(core.tokens(sentence)) > 100:
        flags.append("over_100_tokens")
    if not regex.search(r"[.?!।॥][\"'”’»)\]}]*$", sentence):
        flags.append("no_terminal_punctuation")
    return flags


def main():
    parser = core.parser(__doc__)
    parser.add_argument("--abbreviation", action="append", default=[])
    args = parser.parse_args()
    root = core.setup(args)
    documents, dependencies = core.load_stage(root, "deduplicated")
    included = sorted([d for d in documents if d["included"]], key=lambda d: (d["category"], d["document_id"]))
    rows = list(sentence_rows(included, args.abbreviation))
    review_rows = [{**row, "flags": ";".join(flags)} for row in rows
                   if (flags := segmentation_flags(row["sentence"]))]
    core.write_csv(root / "results/segmentation_review.csv", review_rows, [*SENTENCE_FIELDS, "flags"])
    combined = root / "data/processed/combined"
    outputs = [combined / "documents.jsonl", combined / "konkani_devanagari_clean.txt",
               combined / "sentences.txt", combined / "sentences.tsv"]
    core.atomic_text(outputs[0], core.jsonl_text(included))
    core.atomic_text(outputs[1], "\n\n".join(d["text"] for d in included) + ("\n" if included else ""))
    core.atomic_text(outputs[2], "".join(row["sentence"] + "\n" for row in rows))
    core.write_csv(outputs[3], rows, SENTENCE_FIELDS, delimiter="\t")
    for category in core.CATEGORIES:
        directory = root / "data/processed/by_category" / category
        category_docs = [d for d in included if d["category"] == category]
        text_path, table_path = directory / "corpus_clean.txt", directory / "sentences.tsv"
        core.atomic_text(text_path, "\n\n".join(d["text"] for d in category_docs) + ("\n" if category_docs else ""))
        core.write_csv(table_path, [row for row in rows if row["category"] == category], SENTENCE_FIELDS, delimiter="\t")
        outputs.extend([text_path, table_path])
    counts = {}
    for row in rows:
        counts[row["document_id"]] = counts.get(row["document_id"], 0) + 1
    audit = [{"document_id": d["document_id"], "category": d["category"], "source_file": d["source_file"],
              "extracted_file": d.get("extracted_file", ""), "source_sha256": d.get("sha256", ""),
              "clean_sha256": d.get("clean_sha256", ""), "export_sha256": core.text_hash(d.get("text", "")) if d["included"] else "",
              "processing_status": d["processing_status"], "review_status": d["effective_review_status"],
              "included": d["included"], "provisional": d.get("provisional", False),
              "duplicate_of": d.get("duplicate_of", ""), "duplicate_kind": d.get("duplicate_kind", ""),
              "reason": d.get("duplicate_kind") or d.get("validation_reason", ""),
              "paragraphs_removed": d.get("paragraphs_removed", 0), "raw_tokens": d.get("raw_tokens", 0),
              "cleaned_tokens": d.get("cleaned_tokens", 0),
              "exported_tokens": len(core.tokens(d.get("text", ""))) if d["included"] else 0,
              "sentences": counts.get(d["document_id"], 0)} for d in documents]
    core.write_csv(root / "results/document_audit.csv", audit,
                   ["document_id", "category", "source_file", "processing_status", "included", "reason"])
    for path in outputs:
        dependencies[core.relative(root, path)] = core.sha256_file(path)
    core.save_stage(root, "segmented", documents, dependencies,
                    {"script": "segment_sentences", "abbreviations": args.abbreviation})
    logging.info("Exported %d documents and %d sentences across seven categories", len(included), len(rows))
    if review_rows:
        logging.warning("%d segments need boundary review; see results/segmentation_review.csv", len(review_rows))
    if not included:
        logging.warning("Corpus is empty. Complete manual review, or use validation --include-pending for provisional analysis.")


if __name__ == "__main__":
    core.cli(main)
