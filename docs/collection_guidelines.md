# Collection guidelines

1. Keep one original source per file under the appropriate `data/raw/` category.
   Preserve the original bytes; corrections belong in derived text and review
   records. Prefer `KOK_<category-code>_<number>_<short-title>.<extension>` when
   assigning IDs yourself, and coordinate numbers with `metadata/sources.csv`.
2. Use the seven categories from the project brief. Existing names
   `02_news_and_media` and `06_webarticles` are accepted aliases. Store dictionaries,
   word lists and terminology resources in `reference_only`. Keep gold references
   separate to avoid overlap with the research corpus.
3. Record actual titles, authors/publishers, source URLs, collection dates,
   collectors and permission/license evidence in `metadata/sources.csv`. Extraction
   supplies IDs, paths, input formats and SHA-256; it cannot supply missing facts.
4. Verify that text is Konkani, not merely Devanagari. Sample the start, middle and
   end, and inspect mixed-language paragraphs, front matter and OCR failures.
   Review scripts, languages and category labels independently.
5. After reviewing quality flags, set `review_status=accepted` and
   `language_status=konkani` only for a suitable source. Keep uncertain sources
   pending; reject out-of-scope material with a reason in notes or the exclusion
   register. Language acceptance and redistribution permission are separate.
6. For transcripts, record the source recording, speaker/consent context where
   relevant, transcription method, and any manual corrections. Devanagari-rendered
   English or Hindi can pass script checks and still be inappropriate corpus data.
7. Keep large original and derived files in shared storage or DVC. Do not infer
   reuse permission from public availability or from academic use. Retain source
   metadata even when corpus files cannot be published.

Run commands and processing policies are in [pipeline.md](pipeline.md). Add new
documents incrementally and rerun extraction with the same input scope, followed
by the remaining five stages. Do not merge whole books or newspaper issues into
one raw TXT file; document-level provenance and duplicate detection rely on those
boundaries.
