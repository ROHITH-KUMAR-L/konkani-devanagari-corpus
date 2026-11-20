# Phase 1 pipeline: implementation and usage

The attached brief is the current scope: a traceable Devanagari Konkani corpus,
seven categories, and Phase 1 statistics. Model training, train/test release
splits, harvesting new sources, and the final linguistic research report are not
part of these scripts.

## Repository findings

Before implementation, the six requested files were empty. The repository already
contained working `extract_ldcil.py`, `fetch_epaper.py`, and
`ocr_quality_check.py` helpers, literature/culture OCR text, newspaper PDFs and
cached newspaper OCR, an education PDF with known broken text extracts, and new
public-information, web and transcript text. These originals are left untouched.

The existing raw names `02_news_and_media` and `06_webarticles` are supported as
aliases of `02_news_media` and `06_web_articles`. Both names may exist; each file
is registered separately and duplicate checking happens later. There is no
automatic file renaming or moving. Category names are provenance labels, not a
claim that the supplied content belongs in that genre.

Existing manual exclusions are honored, including the mislabelled duplicate book
and broken education extracts. `reference_only` and `08_ldcil_gold_sample` are
outside the seven automatic input categories. The existing gold-sample extraction
and OCR comparison helpers remain independent. Importing gold material into the
prose corpus requires an explicit collection decision, review and provenance.

## Setup

Python 3.10+ is required. From the project root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The examples below use `python`; use `.venv\Scripts\python.exe` if the environment
is not activated. On Linux/macOS use `.venv/bin/python`. All six scripts accept
`--project-root PATH`, `--verbose`, and `--help`. Relative input/output arguments
are resolved against the project root, so invocation from another working
directory is supported. Tests use temporary projects, not the real corpus.

## Run the six stages

```powershell
python scripts/extract_text.py --extensions txt
python scripts/clean_text.py
python scripts/validate_konkani.py
python scripts/deduplicate.py
python scripts/segment_sentences.py
python scripts/analyze_corpus.py
```

This example processes the currently available TXT sources. Without
`--extensions`, extraction attempts TXT, PDF, DOCX, HTML/HTM, XML and EPUB. A
filtered extraction run describes only that input scope; it does not append
previously processed formats to a new corpus. Keep the same scope when rerunning
an incremental collection, or omit the filter when ready for all formats.

1. **Extract:** discover files, allocate stable IDs, merge the source register,
   compute raw SHA-256, and write `data/raw_txt/<category>/<document_id>.txt`.
   The supported input codecs are UTF-8 (with/without BOM), UTF-16 and UTF-32 with
   BOM; BOM-less UTF-16 has a documented byte-pattern heuristic. Other encodings
   fail explicitly instead of inserting replacement characters. Plain text is
   NFC/LF. DOCX includes body paragraphs and tables in order; XML favors a body
   element; EPUB follows its reading-order spine and omits navigation documents.
2. **Clean:** retain `<id>.minimal.txt` and `<id>.clean.txt` under
   `data/processed/documents/<category>/`; remove HTML fragments and unwanted
   controls, normalize spacing and fix spaces before punctuation. Default rules
   keep numbers, proper nouns, all scripts, combining marks, ZWJ/ZWNJ and line
   structure. No guessed spelling corrections or automatic Hindi/Marathi deletion.
3. **Validate:** create document and paragraph review tables. Only an explicitly
   accepted Konkani document passing the technical checks is eligible by default.
4. **Deduplicate:** exclude identical eligible documents using raw SHA-256 or
   NFC/whitespace-normalized text; retain audit relationships. Accepted copies
   take precedence over provisional copies. Near duplicates are reported for
   review and remain included until a human decides otherwise.
5. **Segment:** write the seven category corpora and sentence TSVs, combined
   documents JSONL, corpus TXT, sentence TXT/TSV and document audit.
6. **Analyze:** write tables and Matplotlib PNGs for combined and category scopes,
   including empty categories. The plots use the noninteractive Agg backend.

The stage files under `data/processed/stages/` are machine-generated JSONL plus
JSON manifests. They retain documents that were excluded, failed extraction, or
remain pending, with reasons. Final combined JSONL contains only eligible,
deduplicated documents. Do not edit generated stage files.

## Review workflow and provenance

Extraction creates/merges `metadata/sources.csv`. Unknown titles, authors, URLs,
collectors, dates and licenses are left blank or `unknown`; filenames are not
invented bibliographic metadata. Complete those fields from real evidence.
`source_file` connects each ID to its original file. Existing custom CSV columns
are retained. Filename IDs such as `KOK_LIT_0001_title.txt` are respected;
otherwise IDs are allocated and persisted in the register. Keep that register
under version control. To rename a source while preserving its ID, update its
`source_file` field before rerunning extraction.

Human review fields:

| Field | Values and meaning |
|---|---|
| `review_status` | `pending`, `accepted`, `rejected`; default `pending` |
| `language_status` | `konkani` (also `kok`/`gom`) after verification; otherwise `unknown`, `mixed`, `hindi`, `marathi`, `english`, etc. |
| `license_status` | Actual rights/permission evidence or `unknown`; independent of language acceptance |
| `notes` | Review evidence, mixed-language passages, OCR problems, genre concerns and other observations |

Set **both** `review_status=accepted` and `language_status=konkani` after reading a
representative sample and resolving flagged passages. A high Devanagari-letter
ratio cannot distinguish Konkani, Hindi, Marathi, or another language transcribed
into that script. The tiny language-marker lists only direct reviewer attention;
they do not provide calibrated language probabilities. Some transcripts contain
English represented in Devanagari, which a script check cannot detect.

`metadata/quality_checks.csv` records effective review/inclusion decisions and
`results/paragraph_review.csv` contains suspect paragraphs. Very low Devanagari
letter ratios, empty text, or substantial replacement-character damage keep a
document out even if accepted. Use `--min-deva` only for a documented policy.
Paragraph flags do not silently remove passages from an otherwise accepted source.
Rejected documents and manually excluded documents never enter a provisional run.

```powershell
# Explicit exploration before review; final statistics will say "Provisional".
python scripts/validate_konkani.py --include-pending
python scripts/deduplicate.py
python scripts/segment_sentences.py
python scripts/analyze_corpus.py
```

Remove `--include-pending` and rerun these stages when review is complete. An empty
final corpus with all documents pending is an intentional outcome, not a reason
to auto-accept the sources. Language acceptance does not grant redistribution
permission; license status stays visible and must be checked before sharing data.

Manual exclusions in `metadata/excluded_documents.csv` use a `file` or
`document_id`, a reason, optional `kept_instead`, and blank/`manual` origin.
Pipeline-generated rows have `validation` or `deduplication` origins and are
refreshed without erasing manual entries. To make an exclusion permanent, add a
manual row or change the source's review decision rather than editing a generated
exclusion row. Rerun extraction after changing manual exclusion rules.

## OCR and existing extracted text

PDF pages are checked separately for a usable Devanagari text layer. Suspicious
pages are flagged `needs_ocr` when OCR is disabled, with page numbers and a nonzero
exit code; a partial PDF is never exported as a complete document. Continue with
the remaining successful records only if you intentionally want that subset.

```powershell
python scripts/extract_text.py --ocr --ocr-lang mar+hin --dpi 300 --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe" --tessdata "C:\path\to\tessdata"
```

Tesseract and the specified traineddata are external prerequisites; pip does not
install them. Only suspect pages are rendered and OCR'd. The page methods, engine
version, languages, DPI and PSM are recorded. `mar+hin` is a script-recognition
configuration, not a claim of Konkani language identification. Blank, sparse, or
English front-matter pages may trigger OCR; the threshold is heuristic. OCR
output still needs language and quality review. No live Tesseract or downloaded
language models are necessary for the unit tests; the OCR engine is mocked there.

Already extracted TXT can be registered using the same extractor:

```powershell
python scripts/extract_text.py --in data/raw_txt --out data/interim/imported_text --extensions txt
```

Use a separate project root for an independent imported-text collection. If
`raw_txt` already contains this pipeline's derived files, do not import those files
again as new originals. To use older newspaper OCR, place **one issue-level TXT
per source** in a staging root with supported category folders and register that
root with `--in` and a different `--out`. Do not point directly at the old e-paper
tree: it contains both issue and page copies. Keep its `epaper_issues.csv` linkage
to the original PDF/URL in the new source metadata/notes. Imported text is recorded
as TXT, never falsely claimed to have fresh OCR provenance or a verified original
PDF hash. Original collection helpers are not run automatically.

## Optional cleaning and deduplication policies

```powershell
python scripts/clean_text.py --drop-page-numbers --repeated-headers --min-line-deva 0.7
python scripts/clean_text.py --rejoin-lines --dehyphenate --remove-pattern "^EXACT BOILERPLATE PATTERN$"
python scripts/deduplicate.py --deduplicate-paragraphs --near-threshold 0.8
```

These are explicit alternatives to the default cleaning/dedup commands. A rerun
replaces the previous derived version; options are not cumulative. Review removal
records in `results/cleaning_changes.jsonl` before accepting an aggressive policy.
Standalone numbers can be verse numbers or table values; repeated lines can be
refrains; Latin text can be a proper name. Headers/footers are removed only from
page edges when the same exact line occurs on at least three pages and 60% of all
pages. This needs preserved form-feed boundaries and cannot reliably fix old TXT
where page boundaries were lost. Line joining applies globally, so process poetry
or transcripts separately if they require distinct policies.

Near-duplicate candidates share sampled hashes of five-token shingles, followed
by full shingle-set Jaccard verification (default >=0.8). Candidate retrieval is
approximate and can miss pairs; documents shorter than 20 tokens are not checked.
No near match is automatically removed. `--skip-near` disables that optional
analysis. Exact paragraph deduplication is opt-in and logs both original paragraph
numbers and the retained paragraph's ID. It may remove legitimate repetition.

## Analysis and segmentation conventions

```powershell
python scripts/analyze_corpus.py --stage extracted
python scripts/analyze_corpus.py --stage cleaned
python scripts/analyze_corpus.py --numbers normalize
python scripts/analyze_corpus.py --no-plots
```

Raw and cleaned iterations go to `results/iterations/extracted/` and
`results/iterations/cleaned/`; final results go to `results/`. `--out` can select
another report directory. Analysis compares identical token rules at each stage;
raw/cleaned iterations are explicitly provisional and have not passed final review
and deduplication. The cleaning log records options and before/after document token
counts for each distinct run without appending duplicate entries on identical reruns.

- Tokens are Unicode letters with combining marks and joiners, optional internal
  hyphens/apostrophes, or numeric runs with decimal/group separators. Punctuation
  is not counted. No lowercasing, stemming, spelling normalization or stopword
  removal occurs. Digits are retained by default; `--numbers normalize` groups
  numeric tokens under `<NUM>`, and `--numbers exclude` omits them from counts.
- Word length uses `regex` extended grapheme clusters (`\X`), with a separate
  codepoint distribution. Type- and token-weighted distributions are both saved.
  Unicode grapheme clusters are not a linguistically validated akshara parser.
  Lengths always use original word forms, including original digits, rather than
  measuring the artificial string `<NUM>`.
- Sentence boundaries use danda, double danda, `.`, `?`, `!`, paragraph breaks
  and trailing text. Decimal points, common abbreviations, initialisms and URLs
  are protected. Ellipses stay inside the current sentence. Closing quotes are
  retained and double-danda verse numbers stay with their verse. These are
  heuristics, especially for abbreviations at the end of a sentence or material
  without punctuation. Add known abbreviations with repeated `--abbreviation`.
  `results/segmentation_review.csv` flags segments over 100 tokens or without
  terminal punctuation. Some existing OCR books put blank lines after each visual
  line, creating fragments; the transcript sample has very little punctuation.
  Review these before treating generated segments as verified sentences. The
  pipeline does not invent punctuation to make their length statistics look better.
- Sentence lengths use the same token policy. Standard deviation is population
  standard deviation; percentiles use linear interpolation. Empty scopes have
  zero statistics and coverage achieved = 0, not a claimed target achievement.
- Frequency ties sort by Unicode word order. Coverage reports 10/25/50/75/90/95/99%.
  Combined types are counted from the combined vocabulary, not summed by category.
  Token contribution above 60% produces a domain-dominance notice.
- Each scope gets sentence-length, word-length, frequency-of-frequencies,
  rank-frequency, cumulative-coverage and vocabulary-growth plots. A seventh
  combined figure compares the category token totals (49 PNGs in total).
  When extreme sentence-length outliers hide the main distribution, the main
  histogram shows lengths through the 99th percentile and an inset shows the full
  distribution. Outliers remain in all statistics. `--no-plots` does not delete
  prior PNGs; `analysis_manifest.json` records whether this run generated plots.

Manually annotate representative paragraphs (including prose, poetry, transcripts,
abbreviations and OCR) before reporting segmentation precision/recall. The tests
demonstrate expected cases; they do not establish accuracy on the real corpus.

## Reruns, failures and reproducibility

Original bytes are never rewritten. Cached extraction is reused only when source,
output, settings and extractor-code hashes agree. Existing unregistered or edited
extracted text is protected even with `--force`; choose another output directory
to re-extract and compare it. Changing a registered source resets its review to
pending. Missing sources are recorded and stale text is not used. Newly collected
files get fresh IDs while previous IDs persist.

Run extraction again after adding/removing inputs, then run the later stages. It
rebuilds the selected input manifest; later stages are deterministic full rebuilds,
not append-only updates. Only extraction is incrementally cached. Changing an
upstream source/artifact/code dependency causes later stages to fail with a
"Stale ... stage" message rather than mix incompatible versions. Editing review
fields in `sources.csv` only requires validation and later stages to be rerun.
Keep a single writer per project root; concurrent pipeline runs are not supported.
Atomic per-file writes avoid truncated artifacts, but multi-file export is not a
transaction; a failed stage must be rerun before its results are used.

Failures have explicit log messages and nonzero exit codes. Extraction catches
per-document errors and records unsuccessful sources, then returns nonzero if any
failed. Empty categories are supported. Unsupported formats and categories are
outside discovery scope; export PDF/DOCX etc. to a supported format if needed.
The final report still needs source evidence, interpretations, manual language and
segmentation evaluation, licensing decisions, and OCR quality measurements.

Large raw/derived data is ignored for new Git files; this does not untrack any
historical committed datasets. Keep datasets in shared storage or DVC and commit
scripts, review metadata, documentation and small test fixtures.

Library references used for implementation: [regex Unicode/grapheme support](https://pypi.org/project/regex/)
and [EbookLib document/spine APIs](https://docs.sourcefabric.org/projects/ebooklib/en/latest/tutorial.html).
