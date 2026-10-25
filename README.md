# Konkani (Devanagari) Text Corpus — NLP Experiential Learning

A clean, documented, plain-text corpus of **Konkani written in Devanagari script** (ISO 639-3 `gom`/`kok`), built for the NLP Experiential Learning (EL) project. Phase 1 builds and analyses the corpus. Phases 2 and 3 build word embeddings, language models, neural networks and NER on top of it, so the cleaner and larger the corpus is now, the better everything later will be.

> **Status (25 Oct 2026):** Phase 1 is in progress. Raw data is partly collected and the repo scaffold exists, but every script, doc, metadata and result file is still an empty placeholder.

---

## Table of contents

1. [Team](#1-team)
2. [Current state of the repository (audit)](#2-current-state-of-the-repository-audit)
3. [Repository layout](#3-repository-layout)
4. [Phase 1: Corpus construction and analysis](#4-phase-1--corpus-construction-and-analysis)
5. [Phase 2: Representations and language models](#5-phase-2--representations-and-language-models)
6. [Phase 3: Neural models and applications](#6-phase-3--neural-models-and-applications)
7. [Konkani-specific challenges](#7-konkani-specific-challenges)
8. [How to run](#8-how-to-run)
9. [Timeline and work split](#9-timeline-and-work-split)
10. [Data sources and licensing](#10-data-sources-and-licensing)

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
| `02_news_and_media` | 20 | PDF (*Bhaangarbhuin* e-paper) | 0 (not extracted) | No usable Devanagari text layer, so **needs OCR** |
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
| 7 | **Newspaper PDFs use non-Unicode fonts** | `pdftotext` returns only Latin/whitespace | OCR page images (Tesseract `hin+mar` / Google Vision / Indic OCR) |
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

---

## 3. Repository layout

```
konkani-devanagari-corpus/
├── data/
│   ├── raw/                       # untouched source files (never edited)
│   │   ├── 01_literature/ … 07_oral_transcripts/
│   │   └── 08_ldcil_gold_sample/  # LDC-IL hand-typed XML sample + metadata .xls
│   ├── interim/                   # extracted / OCR'd plain text, 1 file per source
│   │   └── 08_ldcil_gold_sample/  # KO0000x.txt, one paragraph per line
│   └── processed/
│       ├── by_category/<cat>/corpus_clean.txt   # cleaned text per category
│       ├── by_category/<cat>/sentences.tsv
│       └── combined/
│           ├── konkani_devanagari_clean.txt     # final cleaned corpus (UTF-8, LF)
│           ├── sentences.txt                    # one sentence per line
│           ├── sentences.tsv                    # doc_id, sent_id, n_tokens, sentence
│           └── documents.jsonl                  # {doc_id, category, source, title, text}
├── scripts/
│   ├── extract_text.py        # PDF/DOCX/HTML → text, OCR fallback
│   ├── extract_ldcil.py       # LDC-IL XML → text + metadata CSV  ✅
│   ├── ocr_quality_check.py   # CER/WER of OCR text against the LDC-IL gold sample  ✅
│   ├── clean_text.py          # normalisation + noise removal
│   ├── validate_konkani.py    # script ratio + Konkani-vs-Marathi/Hindi check
│   ├── deduplicate.py         # exact + near-duplicate (MinHash/line Jaccard) removal
│   ├── segment_sentences.py   # paragraph → sentences
│   └── analyze_corpus.py      # type/token, coverage, length analyses + plots
├── metadata/
│   ├── sources.csv            # doc_id, file, title, author, year, category, url, licence
│   ├── excluded_documents.csv # file, reason, kept_instead
│   ├── ldcil_sample_metadata.csv # header fields of the LDC-IL sample files
│   ├── quality_checks.csv     # per-file: tokens, deva %, latin %, FFFD, pass/fail
│   └── data_dictionary.md     # column definitions for every output file
├── results/
│   ├── corpus_statistics.txt
│   ├── word_frequencies.csv   # rank, type, freq, rel_freq, cum_coverage
│   ├── coverage_analysis.csv  # coverage %, types needed
│   ├── document_audit.csv
│   ├── ocr_quality/           # summary.txt, per_paragraph.csv, confusions.csv
│   └── plots/                 # Zipf, coverage, sentence-length, word-length (to add)
├── docs/
│   ├── references/            # LDC-IL Konkani corpus documentation (PDF)
│   ├── collection_guidelines.md
│   ├── data_cleaning_log.md   # every cleaning iteration: what changed, before/after stats
│   └── report.md              # Phase 1 report
├── requirements.txt
└── README.md
```

---

## 4. Phase 1: Corpus construction and analysis

Each step below maps to a point in the assignment brief.

### Step 1–2: Language and team ✅
Konkani, Devanagari script (the official script of Konkani in Goa). Team of four.

### Step 3: Collect and convert to plain text
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

**Working now:**

```bash
pip install -r requirements.txt
python scripts/extract_ldcil.py          # LDC-IL XML → data/interim/08_ldcil_gold_sample/*.txt + metadata CSV
python scripts/ocr_quality_check.py      # CER/WER of our OCR vs LDC-IL gold → results/ocr_quality/
```

**Planned:** the other scripts are still placeholders. The commands below are their intended interface.

```bash
# also install Tesseract + hin/mar traineddata for OCR

python scripts/extract_text.py   --in data/raw --out data/interim
python scripts/clean_text.py     --in data/interim --out data/processed/by_category
python scripts/validate_konkani.py --in data/processed/by_category --report metadata/quality_checks.csv
python scripts/deduplicate.py    --in data/processed/by_category --log metadata/excluded_documents.csv
python scripts/segment_sentences.py --in data/processed/by_category --out data/processed/combined
python scripts/analyze_corpus.py --in data/processed/combined --out results/

file data/processed/combined/*.txt      # must say: UTF-8 text (no CRLF)
```

Planned `requirements.txt`: `pymupdf`, `pdfplumber`, `pytesseract`, `pdf2image`, `beautifulsoup4`, `python-docx`, `regex` (for `\X` grapheme clusters), `indic-nlp-library`, `datasketch`, `pandas`, `matplotlib`, `tqdm`; Phase 2+: `sentencepiece`, `gensim`, `fasttext`, `kenlm`, `torch`, `transformers`, `sklearn-crfsuite`, `seqeval`.

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

Every document is listed in `metadata/sources.csv` with title, author/editor, publisher, year and URL. Book scans and e-paper issues are copyrighted by their publishers (e.g. Sahitya Akademi, *Bhaangarbhuin* / The Goan). They are used here **only for academic, non-commercial research**, and raw copyrighted files should not be redistributed publicly. Consider making the GitHub repo private, or keeping only derived statistics public.
