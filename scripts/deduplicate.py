"""Record raw-byte, normalized-text and near duplicates; never delete source files.

Usage: python scripts/deduplicate.py [--deduplicate-paragraphs] [--near-threshold 0.8]
Near duplicates are review candidates, not automatic exclusions. Exact duplicates
are excluded only against another eligible document; manually accepted text wins.
"""
from __future__ import annotations

import hashlib
import heapq
import logging
from collections import defaultdict

import regex

if __package__:
    from . import corpus_common as core
else:
    import corpus_common as core


def normalized_text(text: str) -> str:
    return regex.sub(r"\s+", " ", core.normalize(text)).strip()


def shingles(text: str, width=5) -> set[int]:
    words = core.tokens(text)
    if not words:
        return set()
    width = min(width, len(words))
    return {int.from_bytes(hashlib.blake2b("\x1f".join(words[i:i + width]).encode("utf-8"),
                                         digest_size=8).digest(), "big")
            for i in range(len(words) - width + 1)}


def jaccard(left: set, right: set) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def find_near_duplicates(documents: list[dict], threshold=.8, sample_size=128) -> list[dict]:
    """Bottom-hash candidates, then exact 5-token-shingle Jaccard verification.

    Candidate retrieval is approximate and can miss pairs. No full all-pairs
    comparison and no automatic deletion based on similarity are performed.
    """
    index = defaultdict(set)
    previous = {}
    matches = []
    for document in documents:
        text = document["text"]
        if len(core.tokens(text)) < 20:
            continue
        current = shingles(text)
        sample = heapq.nsmallest(sample_size, current)
        candidates = set()
        for value in sample:
            candidates.update(index[value])
        for identity in sorted(candidates):
            other, size = previous[identity]
            if min(len(current), size) / max(len(current), size) < threshold:
                continue
            similarity = jaccard(current, shingles(other["text"]))
            if similarity >= threshold:
                matches.append({"document_id": document["document_id"], "duplicate_of": identity,
                                "kind": "near_document", "similarity": similarity,
                                "action": "manual_review", "paragraph_id": "", "kept_paragraph_id": ""})
        identity = document["document_id"]
        previous[identity] = (document, len(current))
        for value in sample:
            index[value].add(identity)
    return matches


def deduplicate_documents(documents, paragraph_dedup=False):
    raw_seen, text_seen, paragraph_seen = {}, {}, {}
    relationships, exclusions = [], []
    # An accepted copy must never lose to an unreviewed or rejected copy.
    ordered = sorted(documents, key=lambda d: (d.get("effective_review_status") != "accepted", d["document_id"]))
    for document in ordered:
        document["duplicate_of"] = ""
        document["duplicate_kind"] = ""
        if not document.get("included"):
            continue
        identity = document["document_id"]
        digest = core.text_hash(normalized_text(document["text"]))
        document["normalized_sha256"] = digest
        duplicate = raw_seen.get(document.get("sha256")) or text_seen.get(digest)
        if duplicate:
            kind = "raw_sha256" if document.get("sha256") in raw_seen else "normalized_text"
            document.update(included=False, duplicate_of=duplicate["document_id"], duplicate_kind=kind)
            relationships.append({"document_id": identity, "duplicate_of": duplicate["document_id"],
                                  "kind": kind, "similarity": 1.0, "action": "excluded",
                                  "paragraph_id": "", "kept_paragraph_id": ""})
            exclusions.append(core.exclusion(document, kind, "deduplication", duplicate["source_file"]))
            continue
        document["paragraphs_removed"] = 0
        if paragraph_dedup:
            kept = []
            for number, paragraph in enumerate(regex.split(r"\n\s*\n", document["text"]), 1):
                key = core.text_hash(normalized_text(paragraph))
                if not paragraph.strip():
                    continue
                if key in paragraph_seen:
                    document["paragraphs_removed"] += 1
                    owner, paragraph_id = paragraph_seen[key]
                    relationships.append({"document_id": identity, "duplicate_of": owner,
                                          "kind": "exact_paragraph", "similarity": 1.0,
                                          "action": "paragraph_removed", "paragraph_id": number,
                                          "kept_paragraph_id": paragraph_id})
                else:
                    paragraph_seen[key] = (identity, number)
                    kept.append(paragraph)
            document["text"] = "\n\n".join(kept)
            if not kept:
                document["included"] = False
                document["duplicate_kind"] = "all_paragraphs_duplicated"
                exclusions.append(core.exclusion(document, "all_paragraphs_duplicated", "deduplication"))
        if document["included"]:
            # Also identify copies of the retained text after paragraph removal.
            text_seen[core.text_hash(normalized_text(document["text"]))] = document
            if document.get("sha256"):
                raw_seen[document["sha256"]] = document
            text_seen[digest] = document
    return documents, relationships, exclusions


def main():
    parser = core.parser(__doc__)
    parser.add_argument("--deduplicate-paragraphs", action="store_true",
                        help="Opt-in; removes repetitions including refrains/dialogue, with paragraph audit")
    parser.add_argument("--near-threshold", type=float, default=.8)
    parser.add_argument("--skip-near", action="store_true")
    args = parser.parse_args()
    root = core.setup(args)
    if not 0 < args.near_threshold <= 1:
        parser.error("--near-threshold must be in (0,1]")
    documents, dependencies = core.load_stage(root, "validated")
    documents, relationships, exclusions = deduplicate_documents(documents, args.deduplicate_paragraphs)
    if not args.skip_near:
        relationships.extend(find_near_duplicates([d for d in documents if d["included"]], args.near_threshold))
    core.write_csv(root / "results/duplicate_relationships.csv", relationships,
                   ["document_id", "duplicate_of", "kind", "similarity", "action", "paragraph_id", "kept_paragraph_id"])
    core.refresh_exclusions(root, "deduplication", exclusions)
    core.save_stage(root, "deduplicated", documents, dependencies,
                    {"script": "deduplicate", "paragraph_dedup": args.deduplicate_paragraphs,
                     "near_threshold": args.near_threshold, "skip_near": args.skip_near})
    logging.info("%d included documents; %d duplicate relationships recorded",
                 sum(d["included"] for d in documents), len(relationships))


if __name__ == "__main__":
    core.cli(main)
