# Project progress

Last updated: **21 November 2026 (Asia/Calcutta)**

Working branch: **`main_2`**

The six Phase 1 scripts are implemented and tested. The main project now has
extracted text for 19 documents, but language review, PDF OCR, full production
processing, and the final research report remain unfinished. The successful
end-to-end sample run is provisional and does not establish a reviewed corpus.

This file records the current implementation and data status. The older progress
checklist and collection estimates in the README describe the October audit.
See [pipeline.md](pipeline.md) for commands and policies and
[the data dictionary](../metadata/data_dictionary.md) for output definitions.

## Completed work

### Project analysis and preservation

- [x] Read the supplied project brief, README, PRD, existing scripts, metadata,
  source directories and earlier collection records.
- [x] Confirm that the six requested pipeline scripts were initially empty.
- [x] Implement the seven-category structure, including aliases for the existing
  `02_news_and_media` and `06_webarticles` directory names.
- [x] Keep reference resources and the LDC-IL gold-sample directory outside
  automatic primary-corpus discovery.
- [x] Honor the existing exclusions for the duplicate literature file and three
  broken education text extracts.
- [x] Preserve original sources; verify all 50 existing raw files by SHA-256
  during extraction/testing without modifying their bytes.
- [x] Retain the existing LDC-IL extractor, newspaper collection/OCR helper and
  OCR quality-comparison helper.

### Script implementation

| Component | Completed behavior |
|---|---|
| [corpus_common.py](../scripts/corpus_common.py) | Shared schemas, Unicode tokenization, strict decoding, hashing, atomic text writes, source-register handling and stale-stage checks |
| [extract_text.py](../scripts/extract_text.py) | Category discovery; TXT, PDF, DOCX, HTML, XML and EPUB extraction; stable IDs; source SHA-256; cached extraction; explicit PDF OCR requirements; page-level OCR provenance |
| [clean_text.py](../scripts/clean_text.py) | NFC/LF normalization, whitespace/punctuation cleanup, markup removal, preserved minimal and analysis versions, optional page-number/header/noise rules and removed-text audit |
| [validate_konkani.py](../scripts/validate_konkani.py) | Encoding/script checks, transparent language-marker hints, paragraph review output, accepted/rejected/pending decisions and explicit provisional inclusion |
| [deduplicate.py](../scripts/deduplicate.py) | Raw SHA-256 and normalized-text document duplicates, optional exact paragraph deduplication, approximate near-duplicate candidates and recorded relationships |
| [segment_sentences.py](../scripts/segment_sentences.py) | Danda/full-stop/question/exclamation boundaries; abbreviation, decimal, initialism, URL and quote handling; verse-number handling; category/combined exports and segmentation review flags |
| [analyze_corpus.py](../scripts/analyze_corpus.py) | Combined/category frequency, TTR, hapax, coverage, sentence-length and grapheme/codepoint word-length analyses; category comparison and Matplotlib plots |

- [x] Preserve Devanagari combining marks, nukta and ZWJ/ZWNJ characters.
- [x] Handle UTF-8 and UTF-16; fail explicitly on unsupported or damaged encoding.
- [x] Keep numbers and proper nouns by default; make stronger cleaning rules opt-in.
- [x] Preserve document provenance through per-document files, stage manifests,
  combined JSONL, sentence IDs and audit outputs.
- [x] Reuse unchanged extraction outputs and retain stable IDs as inputs grow.
- [x] Detect stale upstream inputs rather than silently combining incompatible
  stage results.
- [x] Record failures and duplicate relationships instead of deleting raw sources.
- [x] Require explicit manual Konkani acceptance for default final exports;
  Devanagari script alone does not identify the language.
- [x] Report near duplicates for manual review; do not automatically exclude them.

### Statistical analysis and visualizations

- [x] Implement total tokens, unique types, type-token ratio, hapax counts and
  decreasing-frequency word tables.
- [x] Implement coverage targets of 10%, 25%, 50%, 75%, 90%, 95% and 99%.
- [x] Implement sentence-length minimum, maximum, mean, median, population standard
  deviation and percentiles.
- [x] Implement word-length distributions using Unicode grapheme clusters and
  codepoints, weighted by both types and tokens.
- [x] Implement category document/token totals, vocabulary, sentence lengths,
  contribution ratios and category-dominance notices.
- [x] Generate sentence-length, word-length, frequency-of-frequencies,
  rank-frequency, cumulative-coverage and vocabulary-growth plots for all eight
  scopes, plus a category-token comparison: **49 PNGs** per plotted run.
- [x] Keep extracted/cleaned iteration reports separate from final-stage reports.

### Environment, documentation and Git

- [x] Create the local `.venv` and install the pinned dependencies in
  [requirements.txt](../requirements.txt).
- [x] Pass dependency compatibility, Python syntax and Git whitespace checks.
- [x] Document execution, metadata review, OCR setup, incremental collection,
  cleaning options and known limitations in [pipeline.md](pipeline.md).
- [x] Fill [collection_guidelines.md](collection_guidelines.md) and
  [data_dictionary.md](../metadata/data_dictionary.md).
- [x] Update the README with the implemented commands and current scope.
- [x] Ignore local environments, caches, test outputs and new corpus datasets.
  Previously tracked datasets remain tracked; the ignore rules do not remove them
  from Git history.
- [x] Create `main_2` and push the implementation in **15 separate commits**, with
  exactly one changed or new project file in each commit.

## Testing completed

- [x] **21 automated tests passed** in [test_pipeline.py](../tests/test_pipeline.py).
- [x] Cover Unicode normalization, UTF-16, nukta/joiner preservation, token rules,
  script ratios, conservative cleaning and repeated page headers.
- [x] Cover sentence punctuation, quotes, abbreviations, initials, decimals, URLs,
  ellipses, verse markers, missing spaces and unpunctuated transcripts.
- [x] Cover accepted-copy precedence, normalized duplicates, paragraph removal,
  near-duplicate review and known statistical results.
- [x] Cover HTML/XML, DOCX/EPUB fixtures and PDF extraction with a simulated OCR
  engine. Live Tesseract OCR was not run as part of this implementation testing.
- [x] Run the full pipeline on temporary fixtures, including empty categories,
  pending/rejected decisions, repeat runs, missing/changed inputs, filename-ID
  collisions and protection of existing extracted text.
- [x] Run the full pipeline on copies of real repository data in the ignored
  `.pipeline-test/real-sample/` directory.

### Provisional real-data sample results

The separate sample used eligible existing TXT sources plus one previously
OCR-extracted newspaper issue. It explicitly allowed pending documents and used
the default conservative cleaning policy. Its results are development evidence,
not approved production corpus statistics.

| Metric | Sample result |
|---|---:|
| Exported documents | 20 |
| Word tokens | 455,982 |
| Unique types | 86,626 |
| Generated segments | 71,814 |
| Generated plots | 49 |
| Paragraphs flagged for review | 3,697 |
| Segments flagged for boundary review | 14,454 |

Some OCR books have blank lines after visual lines, which can produce fragments.
The transcript has long stretches without punctuation. The review outputs expose
these cases; segmentation accuracy has not yet been measured on a manually
annotated representative sample.

## Current main-project data status

The extraction stage has been run in the actual project. The later cleaning,
validation, deduplication, segmentation and analysis stages have only been
completed in test projects so far. Production processed/result placeholders are
not final deliverables.

| Category | Registered sources | Extracted TXT | Needs OCR | Existing exclusions |
|---|---:|---:|---:|---:|
| `01_literature` | 10 | 9 | 0 | 1 |
| `02_news_media` | 20 | 0 | 20 | 0 |
| `03_education_science` | 4 | 0 | 1 | 3 |
| `04_culture_history` | 6 | 6 | 0 | 0 |
| `05_public_information` | 1 | 1 | 0 | 0 |
| `06_web_articles` | 2 | 2 | 0 | 0 |
| `07_oral_transcripts` | 1 | 1 | 0 | 0 |
| **Total** | **44** | **19** | **21** | **4** |

- [x] Populate `data/raw_txt/` with **19 per-document TXT files**.
- [x] Populate [sources.csv](../metadata/sources.csv) with **44 source rows**,
  stable IDs, source paths, hashes, formats and extraction outcomes.
- [x] Record **21 PDFs** as `needs_ocr` without exporting partial documents.
- [x] Retain the **4 manual exclusions** without deleting their source files.
- [ ] Complete human language review: all **44 registered rows remain pending**.
- [ ] Fill verified titles, authors/publishers, source URLs, collectors, collection
  dates and rights information. Unknown values have not been invented.

Earlier newspaper OCR remains under `data/interim/` with its own issue metadata.
It has not been automatically imported into the new manifest: issue-level and
page-level copies must not be counted twice. The earlier gold-comparison report
in [results/ocr_quality/summary.txt](../results/ocr_quality/summary.txt) records
**4.77% CER** and **17.86% WER** after punctuation-spacing normalization; these
measure that comparison sample, not every source or the entire corpus.

New `raw_txt` files, test artifacts and generated datasets remain local under
the ignore rules. Clone/sync shared data separately before reproducing the
production run on another computer.

## Remaining Phase 1 work

1. Complete source metadata and licensing evidence, then review each source for
   actual Konkani content, domain suitability and OCR quality.
2. Resolve the 21 PDF OCR requirements using a verified existing extraction or
   local Tesseract with the required language models. Preserve provenance when
   reusing cached OCR, and review the resulting text.
3. Run cleaning on the main project, inspect removed-text records, and select
   appropriate policies for prose, poetry and transcripts.
4. Complete document and paragraph language review; mark accepted Konkani sources
   explicitly and record rejected/mixed material with reasons.
5. Run production deduplication and resolve near-duplicate candidates; audit
   repeated headers, advertisements and boilerplate.
6. Run production segmentation, inspect long/incomplete segments, and evaluate
   boundaries on approximately 200 manually annotated representative paragraphs.
7. Run raw/cleaned/final analyses, compare iterations, record observations in
   `docs/data_cleaning_log.md`, and interpret category balance and vocabulary.
8. Expand category coverage where needed, especially education/science, and
   assess corpus size/balance using reviewed data rather than sample counts.
9. Complete `docs/report.md` and its submission artifact using reproducible final
   statistics, manual evaluation, limitations and source evidence.
10. Freeze a reviewed corpus release and document any later document-level
    train/dev/test split before starting Phase 2.

Phase 2/3 tokenizers, embeddings, language models, POS/NER and other downstream
experiments have not been implemented. Phase 1 corpus quality remains the priority.
