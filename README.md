# Konkani (Devanagari) Text Corpus — NLP Experiential Learning

A clean, documented, plain-text corpus of **Konkani written in Devanagari script** (ISO 639-3 `gom`/`kok`), built for the NLP Experiential Learning (EL) project. Phase 1 builds and analyses the corpus. Phases 2 and 3 build word embeddings, language models, neural networks and NER on top of it, so the cleaner and larger the corpus is now, the better everything later will be.

> **Requirements:** see [PRD.md](PRD.md) for goals, numbered requirements, acceptance criteria and risks.
>
> **Status (20 Nov 2026):** Phase 1 pipeline complete and runnable end to end (`scripts/run_phase1.sh`). Clean corpus at the last run: **3.47M tokens**, 5 domains (live numbers: [docs/report.md](docs/report.md), regenerated on every run). The full e-paper fetch is still running; when it finishes, re-run the pipeline, zip `data/processed` + `data/splits` for the team, then tag `corpus-v1.0`. See [Progress and remaining work](#progress-and-remaining-work).

---

## Table of contents

0. [Progress and remaining work](#progress-and-remaining-work)
1. [Team](#1-team)
2. [Current state of the repository (audit)](#2-current-state-of-the-repository-audit)
3. [Repository layout](#3-repository-layout)
4. [Phase 1: Corpus construction and analysis](#4-phase-1-corpus-construction-and-analysis)
5. [Phase 2: Representations and language models](#5-phase-2-representations-and-language-models)
6. [Phase 3: Neural models and applications](#6-phase-3-neural-models-and-applications)
7. [Konkani-specific challenges](#7-konkani-specific-challenges)
8. [How to run](#8-how-to-run)
9. [Timeline and work split](#9-timeline-and-work-split)
10. [Data sources and licensing](#10-data-sources-and-licensing)

---

## Progress and remaining work

*Last updated: 20 Nov 2026. Tick items here as they are finished.*

| Phase | Status | Summary |
|---|---|---|
| **Phase 1:** corpus building and analysis | 🟢 About 90% | Pipeline, analyses, segmentation, splits and auto-generated report done (3.47M clean tokens at the last run). Remaining: finish e-paper fetch, re-run, team verification, tag v1.0 |
| **Phase 2:** tokenisers, n-gram LMs, embeddings | ⚪ Not started | Starts once corpus v1.0 is frozen |
| **Phase 3:** neural LMs, NER | ⚪ Not started | Starts after Phase 2 |

### Phase 1: done ✅

**Brief steps 1–3: language, team, collection**
- [x] Language, script and team chosen; repo audit; README + [PRD.md](PRD.md)
- [x] LDC-IL gold sample extracted (`extract_ldcil.py`); OCR quality vs gold: CER 4.77%, WER 17.86% (`ocr_quality_check.py`)
- [x] E-paper download + OCR pipeline (`fetch_epaper.py`); works on Linux (system tessdata auto-detected)
- [x] Textbook `fkokb1ps.pdf` re-OCR'd at 300 dpi with new `--force-ocr` (its text layer has broken matras)
- [x] Konkani Wikipedia (gomwiki dump, CC BY-SA) added as `06_web_articles`: 5th domain
- [x] `extract_text.py`: every source → iteration-0 tree (UTF-8, LF), manifest `metadata/documents.csv`
- [x] `metadata/sources.csv` filled from book front matter (some author/year fields still blank)

**Brief steps 4–7: analysis, observations, cleaning iterations**
- [x] `analyze_corpus.py`: tokens, types, TTR, hapax, frequency list, coverage, Zipf/Heaps fits and plots, overall + per category
- [x] Iteration-0 observations → [docs/data_cleaning_log.md](docs/data_cleaning_log.md) (Wikipedia templates, Marathi book, Hindi parallel text, OCR noise, spelling variants)
- [x] Iteration 1 `clean_text.py`: NFC, junk lines, running headers, boilerplate, Latin/other scripts, punctuation spacing, paragraph rejoin across line and page breaks
- [x] Iteration 2 `validate_konkani.py`: Konkani vs Marathi/Hindi language ID, orthography check, numeric tables → `metadata/quality_checks.csv`
- [x] Iteration 3 `deduplicate.py`: exact paragraph dedup (31% removed) + MinHash near-duplicate documents → `metadata/dedup_log.csv`
- [x] Before/after statistics per iteration (`results/iter0..2`, `results/`), per-document audit `results/document_audit.csv`

**Brief steps 8–10, hand-off**
- [x] `segment_sentences.py`: rules for abbreviations, initials, decimals, list numbers, datelines, verse markers → `sentences.txt` / `.tsv`; **held-out F1 0.989** (154 paragraphs; dev set 0.990)
- [x] Language audit extended to 200 paragraphs: 1.0% non-Konkani (target < 2%)
- [x] `length_analysis.py`: sentence length (tokens) and word length (characters, aksharas; token- and type-weighted) with plots
- [x] [docs/report.md](docs/report.md), [metadata/data_dictionary.md](metadata/data_dictionary.md), [docs/collection_guidelines.md](docs/collection_guidelines.md)
- [x] `make_splits.py`: document-level 80/10/10 split → `data/splits/`
- [x] `scripts/run_phase1.sh` runs everything after OCR in about 2 minutes, ending with `build_report.py`, which refills every number in the report and cleaning log from `docs/templates/`
- [x] Domain balance (FR-C6): `make_splits.py` also writes `data/splits/balanced/` (no category above 60%)
- [x] Category 05 tried: Goa government press releases are unreachable (DIP site down, `goa.gov.in` blocks automated access) and in English; documented in the report

### Phase 1: remaining ⏳
- [ ] Let `python scripts/fetch_epaper.py fetch` finish (about 527 issues, resumable), then re-run `scripts/run_phase1.sh` (the report and cleaning log update themselves)
- [ ] Team: verify `data/eval/segmentation_gold.txt`, `segmentation_gold_heldout.txt` and `language_audit_sample.tsv` (all first-pass annotated by Claude)
- [ ] Team: fill the blank author/year fields in `metadata/sources.csv`
- [ ] Newspaper OCR CER: hand-type about 3 e-paper pages as gold (FR-E6)
- [ ] Optional: categories 05/07 if a reachable Konkani source turns up. Anusvara post-correction (FR-N9) deliberately skipped: both forms are often real words
- [x] Sharing decided: processed corpus and splits stay out of GitHub and go to the team as a zip
- [ ] After the final re-run: zip `data/processed` + `data/splits` for the team, then tag `corpus-v1.0`

### Phase 2: remaining (all) ⚪
*Proposed; revise when the official brief arrives. Details in [§5](#5-phase-2-representations-and-language-models).*
- [ ] Tokenisers: whitespace baseline + SentencePiece BPE/Unigram; fertility and OOV rate
- [ ] N-gram LMs (1–5-gram) with Laplace, Good-Turing, Kneser-Ney; test perplexity
- [ ] Word embeddings: word2vec (CBOW, skip-gram), FastText, GloVe
- [ ] Konkani word-similarity and analogy test sets; nearest neighbours; t-SNE/UMAP plots
- [ ] Learning curves with 10/25/50/100% of the corpus
- [ ] Phase 2 report

### Phase 3: remaining (all) ⚪
*Proposed; details in [§6](#6-phase-3-neural-models-and-applications).*
- [ ] Neural LMs: feed-forward → LSTM/GRU → small Transformer; perplexity compared with n-gram models
- [ ] NER annotation guidelines + 2–3k annotated sentences (PER, LOC, ORG, MISC, DATE); Cohen's κ
- [ ] NER models: CRF → BiLSTM-CRF (FastText) → fine-tuned MuRIL/IndicBERT; P/R/F1
- [ ] Optional: POS tagging, domain classification, OCR spell-corrector
- [ ] Final report and presentation

---

## 1. Team

| Member | Role (proposed) |
|---|---|
| Member 1 | Pipeline lead: extraction, cleaning, repo maintenance |
| Member 2 | Data collection: news, web, public information |
| Member 3 | Analysis: type/token, coverage, length plots |
| Member 4 | Linguistic QA: Konkani vs Marathi/Hindi filtering, spelling, sentence rules, report |

Course: NLP, Experiential Learning, Phase 1 → 3.

---

## 2. Current state of the repository (audit)

### 2.1 Raw data collected so far

| Category | Files | Format | Approx. raw tokens* | Notes |
|---|---|---|---|---|
| `01_literature` | 10 | OCR `.txt` | ~356k | Novels, poetry anthology (Sahitya Akademi), drama, tiatr history |
| `02_news_and_media` | 20 PDFs + e-paper download | PDF (*Bhaangarbhuin* e-paper) | **~360k (21 issues OCR'd so far)** | No usable Devanagari text layer, so **needs OCR**. `scripts/fetch_epaper.py` downloads and OCRs the **548 free issues** (Apr 2025 onwards, **~9M tokens** expected), see [§2.4](#24-bhaangarbhuin-e-paper-pipeline) |
| `03_education_science` | 1 PDF + 3 `.txt` | PDF + broken extraction | ~7k | `extracted*.txt` are corrupt (see below); PDF needs OCR |
| `04_culture_history` | 6 | OCR `.txt` | ~248k | History, Mahabharata, story collection, spelling rules |
| `05_public_information` | 0 | n/a | n/a | not started |
| `06_web_articles` | 0 | n/a | n/a | not started |
| `07_oral_transcripts` | 0 | n/a | n/a | not started |
| `08_ldcil_gold_sample` | 5 XML + metadata `.xls` | hand-typed, proofread XML | 15.6k | LDC-IL (CIIL Mysore) gold sample, novels. Clean reference data, see [§2.3](#23-ldc-il-gold-sample-how-we-use-it) |
| **Total** | | | **~627k (~554k after dedup)** | |

\*Whitespace-split tokens on the raw text, before any cleaning.

### 2.2 Problems found in the raw data (to fix in Phase 1)

| # | Observation | Evidence | Planned fix |
|---|---|---|---|
| 1 | **CRLF line endings** in every `.txt` | `file` says "UTF-8 text, with CRLF line terminators" | `dos2unix` / normalise to `\n` |
| 2 | **Duplicate document** | `01_literature/MAHAROOKH.txt` and `04_culture_history/KONKANI KATHA SANGRAH.txt` have identical line sets (Jaccard = 1.0). Both are *Konkani Katha Sangraha*, ed. Chandrakant Keni. The `MAHAROOKH` file is mislabelled | Keep one copy and log the other in `metadata/excluded_documents.csv` |
| 3 | **OCR garbage**: Latin junk lines (`Nore Dell`, `ISTETS`, `OEN`), stray Tamil/Hangul characters (`காணி`, `에`), isolated page numbers | Seen in file heads; 0.1–3.3% Latin characters per file | Line-level filter by Devanagari ratio; drop lines that are mostly digits/Latin |
| 4 | **Front/back matter**: title pages, publisher info, ISBN, © lines, English blurbs | `KONKANI KAVYA SANGRAHA.txt` header | Strip headers and English paragraphs |
| 5 | **HTML remnants** | `<pre>` at start of `GHANAGHAINIYATICHE.txt` (3 tags total) | Strip tags |
| 6 | **Broken PDF extraction** in `03_education_science/extracted*.txt` | 2,006 `U+FFFD` replacement characters, doubled/split vowel signs (`शीारिररीका`) | Discard these and re-OCR `fkokb1ps.pdf` |
| 7 | **Newspaper PDFs use non-Unicode fonts** | `pdftotext` returns only Latin/whitespace; the fonts (ChanakyaUni) have no Unicode mapping | ✅ Render pages and OCR with Tesseract `mar+hin` (`scripts/fetch_epaper.py`) |
| 8 | **Mixed sentence markers** | `।` 12.8k, `॥` 716, `.` 57.8k | Segmenter handles `। ॥ . ? !` and protects abbreviations such as `के.के.` and `पी.ओ.` |
| 9 | **Hyphenated words and line breaks inside paragraphs** from OCR | Short broken lines everywhere | Rejoin lines into paragraphs before segmentation |
| 10 | File name with trailing space: `PALATADANCHE TARUM .txt` | n/a | Rename |
| 11 | **Overlap with the LDC-IL sample** | `KO00002.xml` is the same book as `GHANAGHAINIYATICHE.txt` (Ashok Kamat, 2005) | Keep `KO00002` only as the gold reference for OCR quality and leave it out of corpus counts (logged in `metadata/excluded_documents.csv`) |
| 12 | **Anusvara dropped by OCR** | 118 cases in the gold comparison: `शारांचें` comes out as `शाराचे` | Note in the report. Inflates the type count, because one word appears in two spellings. Candidate for an OCR post-correction pass |

**Tooling available locally:** Python 3.12, `pdftotext`, PyMuPDF, pdfplumber, pytesseract (binary **not installed**), EasyOCR (has no Konkani model, but its `mr`/`hi` models read Devanagari), pandas and matplotlib.

### 2.3 LDC-IL gold sample: how we use it

The **LDC-IL "Gold Standard Konkani Raw Text Corpus"** (CIIL Mysore, 2018) has about 4M hand-typed, proofread words. The full corpus is not available to us, but we do have its free **sample** (`data/raw/08_ldcil_gold_sample/`, 5 novel extracts, 15,595 words) and its **documentation** (`docs/references/LDCIL_Konkani_Raw_Text_Corpus_Documentation_2018.pdf`). We use them in four ways:

1. **Gold reference for OCR quality.** `KO00002` contains hand-typed pages from the same book we have as OCR (`GHANAGHAINIYATICHE.txt`). `scripts/ocr_quality_check.py` fuzzy-aligns each gold paragraph to the OCR text and measures error rates:

   | Metric (41 paragraphs, 12,673 chars, 1,971 words) | Value |
   |---|---|
   | CER, raw OCR | 6.87% |
   | CER, after removing the space OCR adds before punctuation | **4.77%** |
   | WER, raw OCR | 43.33% |
   | WER, after punctuation spacing fix | **17.86%** |

   Main error types (`results/ocr_quality/confusions.csv`): **dropped anusvara `ं`** (118), dropped quotation marks (39), split or merged words (68 space errors), dropped virama `्` (30), dropped vowel sign `ा` (25). **Inference:** the OCR text is usable for counting, but about 1 word in 6 is misspelt. This inflates types and hapax counts, which we expect to see in the Phase 1 type/token analysis.
2. **Clean test data.** Use it to check cleaning rules (cleaning should barely change it) and the sentence segmenter (it uses `.`, `?`, `!` and abbreviations like `आर्.पी.आय.`).
3. **Clean corpus text.** The four non-overlapping samples (`KO00001`, `KO00003–05`, ~13k words) go into the corpus under literature.
4. **Methodology to follow.** Its metadata schema (title, author, publisher, year, sampling, inputter, proofreader) is our template for `metadata/sources.csv`. Its proofreading rules (drop verse inside prose, trim incomplete trailing sentences, remove foreign-script sentences, fix space before punctuation) feed into `clean_text.py`. Its observation that Konkani is *"highly agglutinative… one needs much larger text corpora for good coverage of words"* is a published benchmark to compare our coverage results against.

### 2.4 Bhaangarbhuin e-paper pipeline

*Bhaangarbhuin* is the Konkani daily of The Goan group and our main news source. `scripts/fetch_epaper.py` automates collecting it from [epaper.thegoan.net](https://epaper.thegoan.net/edition/bhaangarbhuin/23246).

| Step | What the script does |
|---|---|
| 1. Find titles | Reads all titles on the site and keeps only **Konkani** ones. Bhaangarbhuin is the only one: Konkansaad and Goan Varta are Marathi; The Goan and the Diwali/Christmas magazines are English |
| 2. List issues | Walks the archive listing newest first: 3,045 issues listed |
| 3. Free issues only | Issues from **2 Apr 2025 onwards (548)** have a free per-page download. Older issues are paid archives (₹29.50) and are **skipped**. robots.txt allows crawling; the script waits 1 s between requests |
| 4. Download | One PDF per page into `data/raw/02_news_and_media/epaper/<title>/<date>_<id>/` (git-ignored, ~2.4 MB per issue) |
| 5. Extract text | PDF text layer if it is real Devanagari; otherwise (always, for this paper) render the page and OCR it with **Tesseract `mar+hin`** (`tessdata_best`) |
| 6. Save | `data/interim/02_news_and_media/epaper/BHAANGARBHUIN/<date>_<id>.txt` (plus per-page files), UTF-8, LF, NFC. One row per issue in `metadata/epaper_issues.csv` |

**Why `mar+hin`:** there is no Tesseract model for Konkani. In our comparison on a sample page, Marathi reads `ळ` correctly (common in Konkani: `वाळपय`, `वेळा`), while Hindi alone produces junk such as `वाव्ठपय`. Combining both gave the best output.

**Results so far:**
- **Yield:** one issue (25 Oct 2026) took 8 pages, about 2 minutes, and gave **16,414 tokens**, with 100% Devanagari letters.
- **Expected total:** 548 issues × ~16k ≈ **9M raw tokens**, before cleaning and deduplication.
- **Known OCR artefacts** for the cleaning step to handle:
  - Masthead and registration-number junk on page 1.
  - Lines from neighbouring columns sometimes interleave.
  - Advertisements.
  - Repeated headers on every issue, such as the paper name and date line, which the dedup step should catch.

**The 20 PDFs downloaded by hand** (`data/raw/02_news_and_media/BHAANGARBHUIN-*.pdf`) are complete 8-page issues from **1–21 Oct 2026**, with 7 Oct missing. File (1) is the newest issue and (20) the oldest. They are not downloaded again. `fetch_epaper.py import` identifies each one by comparing its front page with the site's issues (all 20 matched exactly), then OCRs it into the same `<date>_<issue_id>` layout that `fetch` uses, recording the local file in the `local_pdf` column of `metadata/epaper_issues.csv`. A later `fetch` run sees these issues as done and skips them, so no issue ends up in the corpus twice.

**Status:** 21 issues OCR'd (the 20 local PDFs + 25 Oct 2026) = **360,122 tokens**, 14k–20k per issue. The 20 local PDFs took 77–236 s each.

---

## 3. Repository layout

```
konkani-devanagari-corpus/
├── data/
│   ├── raw/                       # untouched source files (never edited)
│   │   ├── 01_literature/ 02_news_and_media/ 03_education_science/ 04_culture_history/
│   │   ├── 06_web_articles/wikipedia/   # gomwiki dump (CC BY-SA)
│   │   └── 08_ldcil_gold_sample/        # LDC-IL hand-typed XML sample + metadata .xls
│   ├── interim/                   # OCR'd / extracted text (committed)
│   │   ├── 02_news_and_media/epaper/BHAANGARBHUIN/<date>_<id>.txt   # one file per issue
│   │   ├── 03_education_science/  # textbook OCR
│   │   ├── 08_ldcil_gold_sample/  # KO0000x.txt
│   │   └── iter0/ … iter3/        # cleaning iterations (build output, NOT in git)
│   ├── processed/                 # final corpus (build output, NOT in git; shared as a zip)
│   │   ├── by_category/<cat>/{corpus_clean.txt, sentences.tsv}
│   │   └── combined/{konkani_devanagari_clean.txt, sentences.txt, sentences.tsv, documents.jsonl}
│   ├── splits/                    # train/dev/test.txt, balanced/, split_manifest.csv (NOT in git; in the zip)
│   └── eval/                      # segmentation gold, language audit sample
├── scripts/
│   ├── run_phase1.sh          # runs every step below after OCR (~2 min)
│   ├── corpus_utils.py        # shared tokeniser, akshara splitter, document-tree helpers
│   ├── fetch_epaper.py        # e-paper download + OCR; also OCRs any PDF (--force-ocr)
│   ├── extract_ldcil.py       # LDC-IL XML → text + metadata CSV
│   ├── ocr_quality_check.py   # CER/WER of OCR text against the LDC-IL gold sample
│   ├── extract_text.py        # all sources → iteration-0 tree + metadata/documents.csv
│   ├── clean_text.py          # iteration 1: normalisation, junk lines, paragraph rejoin
│   ├── validate_konkani.py    # iteration 2: Konkani vs Marathi/Hindi, orthography, numeric
│   ├── deduplicate.py         # iteration 3: exact paragraph + MinHash document dedup
│   ├── analyze_corpus.py      # type/token, frequency, coverage, Zipf/Heaps + plots
│   ├── document_audit.py      # per-document tokens through every iteration
│   ├── segment_sentences.py   # sentences + final processed files; --eval for P/R/F1
│   ├── length_analysis.py     # sentence and word (chars, aksharas) length distributions
│   ├── make_splits.py         # document-level 80/10/10 split + domain-balanced subset
│   └── build_report.py        # fills docs/templates/*.md with the numbers of the last run
├── metadata/                  # sources.csv, documents.csv, epaper_issues.csv, excluded_documents.csv,
│                              # clean_log.csv, quality_checks.csv, dedup_log.csv, ldcil_sample_metadata.csv, data_dictionary.md
├── results/                   # final stats (top level) + iter0/ iter1/ iter2/, plots/, ocr_quality/
├── docs/                      # report.md + data_cleaning_log.md (generated), templates/, collection_guidelines.md, references/
├── requirements.txt
└── README.md
```

Column-level descriptions of every file: [metadata/data_dictionary.md](metadata/data_dictionary.md).

### Sharing the corpus

`data/processed/`, `data/splits/` and `data/interim/iter*/` are too large for GitHub (hundreds of MB) and contain copyrighted text, so they are **not committed**. Either rebuild them with `scripts/run_phase1.sh` (needs only the committed `data/raw` + `data/interim` text), or use the zip shared within the team:

```bash
zip -r konkani_corpus_processed.zip data/processed data/splits     # sender
unzip konkani_corpus_processed.zip                                  # receiver, from the repo root
```

---

## 4. Phase 1: Corpus construction and analysis

Each step below maps to a point in the assignment brief.

### Step 1–2: Language and team ✅
Konkani, Devanagari script (the official script of Konkani in Goa). Team of four.

### Step 3: Collect and convert to plain text 🟡 (tooling done, collection in progress)
**Target: at least 2M clean tokens**, aiming for 5M+, across at least 5 categories so the corpus is not just literary.

| Source to add | Category | How |
|---|---|---|
| *Bhaangarbhuin* e-paper (more issues) | news | OCR |
| Konkani Wikipedia (`gom.wikipedia.org`, Devanagari pages only) | web | dump + script filter |
| AI4Bharat **IndicCorp v2** / **Sangraha** (Konkani subsets) | web/news | download, then filter and dedupe |
| Goa Govt. Directorate of Official Language, Goa Gazette Konkani notices | public info | scrape + extract |
| Konkani blogs, magazines, *Sunaparant* archives | web | scrape (respect robots.txt) |
| Bhasha/NCERT/SCERT Goa Konkani textbooks | education | OCR |
| Konkani song lyrics, sermons, radio/AIR transcripts | oral | transcribe or collect |

Conversion checklist, applied to every file in `scripts/extract_text.py`:
1. PDF with a valid Unicode text layer → `pdftotext -layout`; otherwise → render pages at 300 dpi → OCR (`tesseract -l hin+mar`, or EasyOCR `['mr']`).
2. HTML/XML → strip tags (BeautifulSoup), drop nav/footer/boilerplate.
3. DOCX → `python-docx`.
4. `iconv -f UTF-16 -t UTF-8` where needed; `dos2unix`; then verify with `file` that every output says **"UTF-8 text"** with no CRLF.
5. Record every file in `metadata/sources.csv`.

### Step 4: Type–token analysis (iteration 0, raw)
`scripts/analyze_corpus.py`:
- **Size** = total tokens; **Vocabulary** = unique types; **TTR** = types/tokens.
- Frequency of each type, sorted in decreasing order → `results/word_frequencies.csv`.
- Hapax legomena count (types with frequency 1), Zipf rank–frequency log-log plot, and Heaps' law plot (vocabulary growth against corpus size).

### Step 5: Coverage analysis
Number of top-frequency types needed to cover **10, 25, 50, 75, 90, 95, 99%** of tokens → `results/coverage_analysis.csv` plus a cumulative coverage curve. Repeat per category to compare literature against news.

### Step 6: Observations and inferences
Record them yourselves in `docs/data_cleaning_log.md` after **every** run. Questions to ask:
- Which top-50 types are punctuation, numbers or OCR garbage?
- Do Marathi/Hindi forms (`आहे`, `आणि`, `है`) appear where Konkani has `आसा`, `आनी`?
- Is the same word counted under several spellings (nukta, chandrabindu vs anusvara, ZWJ/ZWNJ)?
- How does the hapax ratio change after cleaning? A high ratio means OCR errors or rich morphology. Which one is it here?
- Does coverage differ by category? Inflected Konkani needs many types to reach 90% coverage. Quantify it.

### Step 7: Clean and repeat (iterations 1…n)
`scripts/clean_text.py` + `validate_konkani.py` + `deduplicate.py`, re-running Steps 4–5 after each so the report can show **before/after tables**.

| Cleaning rule | Detail |
|---|---|
| Unicode normalisation | NFC; unify nukta forms (`ज़` precomposed vs `ज`+`़`); remove ZWJ/ZWNJ except where they matter; standardise `ऍ ऑ` |
| Character whitelist | Devanagari block U+0900–U+097F, Devanagari digits, `। ॥ . , ? ! ; : - ' "` and whitespace |
| Line filter | Drop lines with Devanagari ratio below 0.7, lines under 3 tokens that are page numbers/headers, and repeated running headers |
| Numbers | Map Devanagari and ASCII digits to a `<NUM>` token for analysis (keep the original in the corpus) |
| Punctuation | Separate it from words before counting types |
| Foreign words | Drop English words/sentences; tag remaining Latin tokens |
| Language ID | Score each paragraph for Konkani vs Marathi/Hindi using marker words (`आसा, ना, आनी, हांव, म्हजो, तुमी, जाल्यार` vs `आहे, आणि, मी, माझा, है, और`). Drop non-Konkani paragraphs |
| OCR errors | Drop types with impossible sequences (vowel sign at word start, doubled matras); later, optionally fix spellings with a frequency-based edit-distance pass |
| Dedup | Exact hash on normalised paragraphs, plus near-duplicates (line Jaccard / MinHash above 0.8) at document level |
| Paragraph rejoin | Merge OCR line-wrapped lines; rejoin hyphenated breaks |

### Step 8: Sentence segmentation
`scripts/segment_sentences.py`, rule-based:
- Boundaries: `।`, `॥`, `?`, `!`, and `.` **only** when it is not part of an abbreviation (`के.के.`, `डॉ.`, `पी.ओ.`, `श्री.`, single-letter initials), a number (`3.5`), or an ellipsis.
- Handle verse/poetry separately: lines ending in `॥ १ ॥`-style verse numbers.
- Output: `sentences.txt` (one sentence per line) and `sentences.tsv`.
- Evaluate it: hand-annotate about 200 paragraphs and report segmentation precision and recall.

### Step 9: Sentence-length and word-length analysis
- Histogram plus summary stats (mean, median, std, percentiles) of sentence length in tokens, overall and per category.
- Word length in **characters and in aksharas (grapheme clusters)**. Akshara length suits Devanagari better, because `क्ष` is one akshara but three code points. Report both, token-weighted and type-weighted.
- Plots saved in `results/plots/`.

### Step 10: Report
`docs/report.md` (then export to PDF): sources → extraction → every cleaning iteration with before/after numbers → final statistics → coverage → length analyses → observations and inferences → limitations → what Phase 2 needs.

**Phase 1 deliverables checklist**
- [ ] ≥ 2M clean tokens, ≥ 5 categories, `sources.csv` complete
- [ ] All final files pass `file` → "UTF-8 text", LF only
- [ ] Raw vs cleaned type/token + frequency list
- [ ] Coverage table + curve (overall and per category)
- [ ] Zipf and Heaps plots
- [ ] Sentences one per line + segmentation evaluation
- [ ] Sentence-length and word-length plots
- [ ] Cleaning log with observations per iteration
- [ ] Final report

---

## 5. Phase 2: Representations and language models

> The brief only says the next phases cover "word embeddings, Language Models, Neural Networks, Named Entities etc." The split between Phase 2 and Phase 3 below is our proposal; adjust it when the official brief arrives.

| Task | Plan | Evaluation |
|---|---|---|
| **Tokenisation** | Whitespace + punctuation baseline; **SentencePiece/BPE** (8k–32k vocab) trained on our corpus, which matters for agglutinative Konkani | Fertility (subwords per word), OOV rate |
| **N-gram language models** | Unigram → trigram (and 4/5-gram via KenLM) with Laplace, Good-Turing, Kneser-Ney smoothing | **Perplexity** on a held-out test split (split by document, not by sentence, to avoid leakage) |
| **Word embeddings** | word2vec (CBOW + skip-gram), **FastText** (subword-aware, so it handles OCR noise and inflection), GloVe | Nearest neighbours, a small hand-made **Konkani analogy and similarity set** (family, gender, place names), t-SNE/UMAP plots |
| **Compare with pretrained** | IndicBERT / MuRIL / IndicTrans tokenisers, FastText `gom` vectors | Same intrinsic tests |
| **Corpus growth study** | Train embeddings/LMs on 10/25/50/100% of the corpus | Perplexity and similarity scores against corpus size; this links back to the Phase 1 coverage analysis |

Data prep for Phase 2 (done at the end of Phase 1): fixed train/dev/test split by document (80/10/10), stored as `data/splits/`.

---

## 6. Phase 3: Neural models and applications

| Task | Plan | Evaluation |
|---|---|---|
| **Neural LM** | Feed-forward (Bengio) → LSTM/GRU → small Transformer (GPT-style) trained from scratch on our corpus | Perplexity against the Phase 2 n-gram models; sample generations |
| **Fine-tune multilingual models** | Continue pretraining IndicBERT/MuRIL on the Konkani corpus (MLM) | Perplexity / downstream scores before and after |
| **Named Entity Recognition** | Build a **gold NER dataset**: annotate about 2–3k sentences (PER, LOC, ORG, MISC, DATE) in Doccano/Label Studio, using inter-annotator agreement (Cohen's κ) across the 4 members. Models: CRF baseline → BiLSTM-CRF with our FastText → fine-tuned MuRIL/IndicBERT | Entity-level P/R/F1 (seqeval) |
| **POS tagging (optional)** | Small annotated set using the BIS tagset; same model ladder as NER | Accuracy |
| **Text classification (optional)** | Use our 7 categories as labels: literature vs news vs history… | Macro-F1 |
| **Spell-checker / OCR post-corrector (stretch)** | Frequency list from Phase 1 + edit distance + LM ranking | Correction accuracy on a hand-corrected sample |

**Final deliverables:** released clean corpus + datasheet, trained embeddings, LMs, NER dataset + models, and a final report/presentation.

---

## 7. Konkani-specific challenges

- **Language contamination:** Konkani shares Devanagari with Marathi and Hindi, and many Goan sources mix them. Script-based filtering is not enough, so word-level language ID is required.
- **Multiple scripts:** Konkani is also written in Roman (Romi), Kannada, Malayalam and Perso-Arabic. This corpus is **Devanagari only**, so Romi Konkani lines are removed, not transliterated.
- **Spelling variation:** dialects (Antruzi, Bardeskari, Saxtti, Mangalorean, Kerala Konkani such as `DASHARATHA.txt`) and pre-standard spellings. `KONKANI SHUDDLEKHANACHE NEM.txt` (spelling rules) can guide normalisation.
- **Rich morphology:** postpositions attached to words (`घरांत`, `घराक`, `घराचो`) inflate the type count, so coverage curves will rise slowly. Expect this, and explain it in the report.
- **Low-resource OCR:** no Tesseract model exists for `kok`, so use `hin`/`mar` and measure character error rate (CER) on a few hand-typed pages.

---

## 8. How to run

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# OCR needs Tesseract 5 with mar + hin traineddata
#   Fedora: sudo dnf install tesseract tesseract-langpack-mar tesseract-langpack-hin
#   Windows: winget install UB-Mannheim.TesseractOCR, traineddata in ~/tessdata

# 1. Slow, cached collection steps (run once)
python scripts/extract_ldcil.py                   # LDC-IL XML → data/interim/08_ldcil_gold_sample
python scripts/ocr_quality_check.py               # OCR CER/WER vs LDC-IL gold → results/ocr_quality/
python scripts/fetch_epaper.py fetch              # all free Bhaangarbhuin issues (~18 h, resumable)
python scripts/fetch_epaper.py ocr --pdf-dir data/raw/03_education_science \
    --out data/interim/03_education_science --dpi 300 --force-ocr
curl -o data/raw/06_web_articles/wikipedia/gomwiki-latest-pages-articles.xml.bz2 \
    https://dumps.wikimedia.org/gomwiki/latest/gomwiki-latest-pages-articles.xml.bz2

# 2. Everything else: cleaning iterations, analyses, segmentation, lengths, splits, report (~2 min)
scripts/run_phase1.sh                             # PYTHON=.venv/bin/python scripts/run_phase1.sh

file data/processed/combined/*.txt                # must say: UTF-8 text (no CRLF)
```

`fetch_epaper.py` keeps only Konkani titles, downloads only issues the site offers free, waits 1 s between requests, and git-ignores the page PDFs. `data/interim/iter*`, `data/processed/` and `data/splits/` are build outputs and are not committed.

---

## 9. Timeline and work split

| Week | Work | Owner |
|---|---|---|
| 1 | Fix audit issues (dedup, rename), write `sources.csv`, set up Tesseract, OCR newspapers + textbook | Pipeline + Collection |
| 1–2 | Collect web/public-info/Wikipedia/IndicCorp data → reach 2M+ tokens | Collection |
| 2 | Iteration 0 analysis on raw text and record observations | Analysis |
| 2–3 | Cleaning + language ID + dedup; iterations 1…n with before/after stats | Pipeline + QA |
| 3 | Sentence segmentation + evaluation; length analyses + plots | QA + Analysis |
| 4 | Report, final checks, freeze corpus v1.0 + train/dev/test splits | All |
| Phase 2 | Tokenisers, n-gram LMs, embeddings | All |
| Phase 3 | Neural LMs, NER dataset + models | All |

---

## 10. Data sources and licensing

**Reference corpus:** Varik, S., Rajesha N., Manasa G., Choudhary, N. & Ramamoorthy, L. (2018). *Konkani Raw Text Corpus* (documentation of LDC-IL's "A Gold Standard Konkani Raw Text Corpus"). Linguistic Data Consortium for Indian Languages, CIIL, Mysore. ISBN 978-81-7343-232-3. We use only the publicly downloadable sample and documentation; © CIIL.

Every document is listed in `metadata/sources.csv` with title, author/editor, publisher, year and URL. Book scans and e-paper issues are copyrighted by their publishers (e.g. Sahitya Akademi, *Bhaangarbhuin* / The Goan). They are used here **only for academic, non-commercial research**, and raw copyrighted files should not be redistributed publicly. The processed corpus is therefore kept out of GitHub and shared only inside the team as a zip (see [Sharing the corpus](#sharing-the-corpus)). The repo still holds OCR text in `data/interim`; consider making it private.
