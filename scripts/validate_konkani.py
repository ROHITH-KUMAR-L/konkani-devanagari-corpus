"""Audit script, encoding and language hints; apply explicit human review decisions.

Usage: python scripts/validate_konkani.py [--include-pending]
Set review_status=accepted and language_status=konkani in metadata/sources.csv
after reviewing a document. Marker words are hints, never a language classifier.
"""
from __future__ import annotations

import logging
import unicodedata
from collections import Counter

import regex

if __package__:
    from . import corpus_common as core
else:
    import corpus_common as core

# Small transparent review lexicons, not statistically trained or calibrated.
MARKERS = {
    "konkani": {"आसा", "आनी", "हांव", "म्हाका", "म्हजो", "जाल्यार", "ताका", "तुमकां"},
    "marathi": {"आहे", "आहेत", "आणि", "माझा", "माझे", "मला", "तुम्ही", "त्यांनी"},
    "hindi": {"है", "हैं", "और", "में", "नहीं", "मुझे", "लेकिन", "उन्होंने"},
}
QUALITY_FIELDS = ["document_id", "category", "source_file", "processing_status",
                  "review_status", "language_status", "effective_review_status", "included",
                  "tokens", "letters", "devanagari_ratio", "latin_ratio", "other_script_ratio",
                  "replacement_characters", "suspicious_marks", "utf8", "lf", "nfc",
                  "konkani_markers", "marathi_markers", "hindi_markers", "flags", "reason"]


def inspect_text(text: str) -> dict:
    counts = Counter(core.tokens(text))
    stats = core.script_stats(text)
    stats.update(tokens=sum(counts.values()), replacement_characters=text.count("\ufffd"),
                 suspicious_marks=len(regex.findall(r"(?<![\p{L}\p{M}\u200c\u200d])\p{M}|([\u093e-\u094c])\1", text)))
    for language, markers in MARKERS.items():
        stats[f"{language}_markers"] = sum(counts[word] for word in markers)
    flags = []
    if stats["replacement_characters"]:
        flags.append("replacement_characters")
    if stats["suspicious_marks"]:
        flags.append("possible_ocr_combining_mark_error")
    foreign = max(stats["marathi_markers"], stats["hindi_markers"])
    if foreign >= 3 and foreign > 2 * stats["konkani_markers"]:
        flags.append("possible_hindi_or_marathi")
    if not stats["konkani_markers"]:
        flags.append("no_konkani_markers_in_small_lexicon")
    stats["flags"] = ";".join(flags)
    return stats


def main():
    parser = core.parser(__doc__)
    parser.add_argument("--min-deva", type=float, default=.7,
                        help="Minimum fraction of LETTERS in Devanagari; does not identify language")
    parser.add_argument("--include-pending", action="store_true", help="Explicit provisional corpus; still excludes rejected/broken text")
    args = parser.parse_args()
    root = core.setup(args)
    if not 0 <= args.min_deva <= 1:
        parser.error("--min-deva must be in [0,1]")
    documents, dependencies = core.load_stage(root, "cleaned")
    registry = {row["document_id"]: row for row in core.load_sources(root)}
    registry_path = root / "metadata/sources.csv"
    dependencies[core.relative(root, registry_path)] = core.sha256_file(registry_path)
    exclusions = core.manual_exclusions(root)
    reports, paragraphs, excluded = [], [], []
    for document in documents:
        current = registry.get(document["document_id"], {})
        # User descriptive/review fields can be updated without re-extracting text.
        for key in core.SOURCE_FIELDS:
            if key not in {"source_file", "extracted_file", "sha256", "processing_status", "document_id", "category"} and key in current:
                document[key] = current[key]
        review = (document.get("review_status") or "pending").lower().strip()
        language = (document.get("language_status") or "unknown").lower().strip()
        if review not in {"accepted", "rejected", "pending"}:
            raise ValueError(f"Invalid review_status {review!r} for {document['document_id']}")
        text = document.get("text", "")
        stats = inspect_text(text)
        data = core.resolve(root, document["extracted_file"]).read_bytes() if document["processing_status"] == "cleaned" else b""
        try:
            decoded = data.decode("utf-8")
            utf8 = True
        except UnicodeError:
            decoded, utf8 = "", False
        stats.update(utf8=utf8, lf=b"\r" not in data, nfc=unicodedata.is_normalized("NFC", decoded))
        manual = exclusions.get(document["document_id"]) or exclusions.get(document["source_file"])
        technically_usable = bool(text.strip() and stats["letters"] and utf8
                                  and stats["devanagari_ratio"] >= args.min_deva
                                  and stats["replacement_characters"] / max(len(text), 1) <= .01)
        reason = ""
        if manual:
            effective, reason = "rejected", manual["reason"]
        elif document["processing_status"] != "cleaned":
            effective, reason = "pending", document.get("error", document.get("exclusion_reason", document["processing_status"]))
        elif review == "rejected":
            effective, reason = "rejected", "manually_rejected"
        elif language in {"hindi", "marathi", "english", "hi", "mr", "en", "non_konkani", "reference_only"}:
            effective, reason = "rejected", "registered_outside_corpus_language_or_scope"
        elif not technically_usable:
            effective, reason = "pending", "empty_low_script_ratio_or_encoding_damage"
        elif review == "accepted" and language in {"konkani", "kok", "gom"}:
            effective = "accepted"
        else:
            effective, reason = "pending", "manual_konkani_confirmation_required"
        included = technically_usable and (effective == "accepted" or (effective == "pending" and args.include_pending))
        document.update(effective_review_status=effective, included=included,
                        validation_reason=reason, quality=stats,
                        provisional=included and effective != "accepted")
        reports.append({**{key: document.get(key, "") for key in QUALITY_FIELDS if key in document},
                        **stats, "review_status": review, "language_status": language,
                        "effective_review_status": effective, "included": included, "reason": reason})
        if not included:
            excluded.append(core.exclusion(document, reason or "not_included", "validation"))
        for number, paragraph in enumerate(regex.split(r"\n\s*\n", text), 1):
            if not paragraph.strip():
                continue
            para_stats = inspect_text(paragraph)
            if para_stats["devanagari_ratio"] < args.min_deva or "possible_hindi_or_marathi" in para_stats["flags"] or para_stats["replacement_characters"] or para_stats["suspicious_marks"]:
                paragraphs.append({"document_id": document["document_id"], "paragraph_id": number,
                                   **para_stats, "text": paragraph})
    core.write_csv(root / "metadata/quality_checks.csv", reports, QUALITY_FIELDS)
    core.write_csv(root / "results/paragraph_review.csv", paragraphs,
                   ["document_id", "paragraph_id", "devanagari_ratio", "flags", "text"])
    core.refresh_exclusions(root, "validation", excluded)
    core.save_stage(root, "validated", documents, dependencies,
                    {"script": "validate_konkani", "min_deva": args.min_deva, "include_pending": args.include_pending})
    logging.info("Included %d/%d documents (%d provisional); %d paragraphs flagged for review",
                 sum(d["included"] for d in documents), len(documents),
                 sum(d["provisional"] for d in documents), len(paragraphs))


if __name__ == "__main__":
    core.cli(main)
