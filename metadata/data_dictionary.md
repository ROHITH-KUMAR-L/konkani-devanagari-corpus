# Corpus data dictionary

Text output is UTF-8, NFC and LF. CSV/TSV uses a header row and standard CSV
quoting; read it with a CSV parser, not by splitting lines on commas. Booleans in
CSV are `True`/`False`; JSON uses native booleans. Ratios are fractions from 0 to 1.
Paths are relative to the project root, unless an external input path was used.

## Source register: `metadata/sources.csv`

| Column | Meaning |
|---|---|
| document_id | Persistent ID, e.g. `KOK_LIT_0001`; unique in the register |
| category | One of the seven canonical directory names |
| subcategory | Human-supplied finer genre |
| title | Verified source title; blank until known |
| author_publisher | Verified authorship/publisher information |
| source_url | Original source URL, if known |
| source_format | Detected input suffix, such as `txt`, `pdf`, `xml` |
| script | Human-recorded script description; script ratios are measured separately |
| language_status | `unknown`, `konkani`/`kok`/`gom`, `mixed`, or an identified other language |
| license_status | Recorded rights evidence or `unknown`; public access is not permission |
| collected_by | Collector, supplied by the team |
| collection_date | Actual collection date, preferably YYYY-MM-DD; never guessed |
| sha256 | SHA-256 of original file bytes |
| processing_status | Latest intake state: `extracted`, `excluded`, `needs_ocr`, `error`, `missing_source` |
| review_status | Human decision: `pending`, `accepted`, `rejected` |
| notes | Provenance, review and rights observations |
| source_file | Path to the untouched source, used to preserve ID assignments |
| extracted_file | Path to this pipeline's per-document extracted TXT |

Legacy `doc_id` and `file` columns are read as fallbacks; additional custom columns
are preserved. `processing_status` in this register describes intake, not progress
through every derived stage. A source-byte change resets `review_status` to pending.

## Quality and exclusions

`metadata/quality_checks.csv` includes document/category/source, intake status,
human review/language states, `effective_review_status`, `included`, `reason`,
`tokens`, `letters`, `devanagari_ratio`, `latin_ratio`, `other_script_ratio`,
`replacement_characters`, `suspicious_marks`, `utf8`, `lf`, `nfc`, three language
marker counts, and semicolon-separated `flags`. Script ratios use letters only:
digits, punctuation and combining marks do not enter the denominator. Marker
counts are heuristic observations, not language probabilities.

`results/paragraph_review.csv` adds one-based `paragraph_id` and the paragraph
`text` to the quality metrics for flagged paragraphs. Paragraphs follow blank-line
boundaries; low-quality OCR may not retain meaningful paragraph boundaries.

`metadata/excluded_documents.csv`: `file`, `reason`, `kept_instead`, optional
`document_id`, and `origin`. Blank/`manual` origins are human decisions and are
preserved. `validation` and `deduplication` rows are generated and refreshed by
their respective stage. Pending sources are listed here because they are withheld
from the current export; that does not mark them permanently rejected.

## Stage records and text files

`data/processed/stages/{extracted,cleaned,validated,deduplicated,segmented}.jsonl`
has one object per discovered document, including failures and exclusions.
`*.meta.json` records options, hashes of dependencies, and the JSONL output hash.
The `@manual_exclusions` dependency hashes only human exclusion decisions.

Stage objects retain source-register fields and add fields as they become available:

- Extraction: `extraction` (method and encoding; PDF page methods, OCR engine,
  version, languages, DPI, PSM), `extracted_sha256`, `extractor_hash`, `error`,
  `exclusion_reason`.
- Cleaning: `text`, `minimal_file`, `clean_file`, `clean_sha256`, `raw_tokens`,
  `cleaned_tokens`, `removed_lines`. `clean_sha256` hashes the analysis string
  before deduplication, without the standalone file's trailing newline.
- Validation: `quality`, `effective_review_status`, `included`,
  `validation_reason`, `provisional`.
- Deduplication: `normalized_sha256`, `duplicate_of`, `duplicate_kind`, and
  optionally a paragraph-deduplicated `text`.

`data/raw_txt/<category>/<id>.txt` contains extracted text. Minimal and analysis
versions are under `data/processed/documents/<category>/<id>.{minimal,clean}.txt`.
These per-document cleaned files are preserved before duplicate-paragraph removal.

`data/processed/combined/documents.jsonl` contains only included final document
records; it retains both text and provenance. `konkani_devanagari_clean.txt` and
the category `corpus_clean.txt` are readable exports separated by blank lines;
use JSONL for reliable document boundaries.

`sentences.tsv` (combined and per category): `document_id`, stable ordered
`sentence_id` (`<id>_S000001`), `category`, `sentence`, `n_tokens`.
`sentences.txt` is one sentence per line. IDs are deterministic for an unchanged
document and segmentation policy; editing text can renumber later sentences.
`n_tokens` uses the default number-preserving policy, even if a later analysis uses
`--numbers normalize` or `exclude`.

## Audit and analysis outputs

| File | Columns / contents |
|---|---|
| `results/document_audit.csv` | All source records: IDs, paths, source/clean/export hashes, intake/review states, inclusion/provisional flags, duplicate relationship/reason, raw/clean/export token counts and sentence counts |
| `results/duplicate_relationships.csv` | `document_id`, `duplicate_of`, `kind`, `similarity`, `action`, `paragraph_id`, `kept_paragraph_id`; paragraph IDs refer to the original cleaned version |
| `results/cleaning_changes.jsonl` | Removed lines with `document_id`, one-based `page`/`line`, `reason`, exact removed `text` |
| `results/segmentation_review.csv` | Sentence TSV columns plus `flags` for long segments or missing terminal punctuation; no automatic boundary corrections |
| `results/word_frequencies.csv` | `scope`, `rank`, `word`, `frequency`, `relative_frequency`, `cumulative_coverage`; descending frequency then Unicode lexical order |
| `results/coverage_analysis.csv` | `scope`, `target_coverage`, `types_required`, `achieved_coverage`, `total_tokens` |
| `results/category_comparison.csv` | `scope`, `documents`, `tokens`, `types`, `type_token_ratio`, `hapax_legomena`, `mean_sentence_length`, `sentences`, `provisional_documents`, `token_contribution` |
| `results/sentence_length_statistics.csv` | `scope`, `sentences`, `min`, `max`, `mean`, `median`, `std` (population), `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` |
| `results/word_length_distribution.csv` | `scope`, `unit` (`graphemes`/`codepoints`), `length`, `type_frequency`, `token_frequency` |
| `results/corpus_statistics.txt` | Token policy, provisional status, scope table, top words and category-dominance notice |
| `results/analysis_manifest.json` | Analysis stage/number policy, provisional flag, input dependency hashes and analysis-library versions |
| `results/plots/<scope>/*.png` | Six plots per scope; category-token comparison is at the plots root |

`scope` is `combined` or one of the seven category names. The combined vocabulary
is a union, not the sum of category type counts. Coverage values are fractions,
not percentages. An empty scope has zero counts and achieved coverage zero.
Raw/cleaned iteration analyses use the same schemas under `results/iterations/`.
See [the pipeline guide](../docs/pipeline.md) for token rules and limitations.
