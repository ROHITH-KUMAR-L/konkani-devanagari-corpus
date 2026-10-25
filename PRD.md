# Product Requirements Document (PRD)
## Konkani (Devanagari) Corpus and NLP Toolkit: NLP Experiential Learning, Phases 1–3

| | |
|---|---|
| **Document owner** | Team (Members 1–4) |
| **Stakeholders** | Course instructor (evaluator), team members, future users of the corpus |
| **Status** | Draft v1.0, Phase 1 in progress |
| **Last updated** | 25 Oct 2026 |
| **Related docs** | [README.md](README.md) (plan + audit), [docs/report.md](docs/report.md) (Phase 1 report), [docs/data_cleaning_log.md](docs/data_cleaning_log.md), [LDC-IL documentation](docs/references/LDCIL_Konkani_Raw_Text_Corpus_Documentation_2018.pdf) |

---

## 1. Summary

We are building a **large, clean, documented plain-text corpus of Konkani in Devanagari script**. On top of it we will build a reproducible pipeline for corpus statistics (Phase 1), word representations and language models (Phase 2), and neural models including Named Entity Recognition (Phase 3). The Phase 1 corpus is the base for everything after it. Its size and cleanliness directly limit the quality of the Phase 2 and 3 results, so Phase 1 requirements carry the most detail here.

## 2. Background and problem

- Konkani is the official language of Goa and a scheduled language of India, with about 2.3M speakers. It is **low-resource** for NLP: there is little clean digital text, no Konkani OCR model, and few labelled datasets.
- It is written in five scripts (Devanagari, Roman, Kannada, Malayalam, Perso-Arabic). Devanagari is the official one. It shares Devanagari with **Marathi and Hindi**, so script-based filtering alone cannot isolate Konkani.
- It is **morphologically rich and agglutinative**. Postpositions attach to words (`घरांत`, `घराक`, `घराचो`), so new word types keep appearing as the corpus grows, and large corpora are needed for good coverage. LDC-IL (2018) reports the same.
- **Our current raw data** (audit, 25 Oct 2026):
  - About 612k raw tokens of OCR text from 18 books, plus 20 newspaper PDFs and 1 textbook PDF that have no usable Unicode text layer.
  - Measured OCR quality against the LDC-IL gold sample: **CER 4.77%, WER 17.86%**.
  - The text contains duplicates, front matter, Latin and other-script junk, CRLF line endings, and broken extractions. See README §2.

**The problem:** without systematic collection, cleaning, Konkani-specific filtering and documentation, every count (types, coverage, lengths) and every Phase 2 and 3 model would be distorted by noise and non-Konkani text.

## 3. Goals and non-goals

### 3.1 Goals
| ID | Goal |
|---|---|
| G1 | Build a Konkani-Devanagari corpus of **≥ 2M clean tokens** (stretch: 5M) covering **≥ 5 domains** |
| G2 | Make it **plain UTF-8 text, LF line endings**, which `file` reports as "UTF-8 text" |
| G3 | Perform and document **type–token, frequency, coverage, sentence-length and word-length analyses**, with observations and inferences from the team |
| G4 | Make the pipeline **reproducible**: raw → clean → analysis with scripts, and every exclusion logged |
| G5 | Deliver a fixed **train/dev/test split** ready for Phase 2 |
| G6 | Phase 2: deliver **tokenisers, n-gram LMs and word embeddings** with intrinsic evaluation |
| G7 | Phase 3: deliver **neural LMs and a Konkani NER dataset + models** with standard metrics |

### 3.2 Non-goals
- Roman (Romi), Kannada or Malayalam-script Konkani. These are **removed**, not transliterated.
- Building a new OCR engine. We use existing engines and measure their error rate.
- Redistributing copyrighted source texts publicly.
- Speech data and audio processing. Oral transcripts are text only.
- Production deployment or a user-facing app.

## 4. Users and use cases

| User | Need | Use case |
|---|---|---|
| Course instructor | Verify rigour of the experiment, observations and inferences | Read the report, re-run the scripts, inspect the statistics |
| Team members | Shared, versioned, clean data | Run the pipeline, add sources, train models |
| Phase 2/3 models | Large, clean, deduplicated, split text | Embedding/LM training, NER annotation pool |
| Future researchers | Documented Konkani resource | Reuse the corpus statistics, frequency list, NER data |

---

## 5. Scope and requirements: Phase 1 (Corpus)

Priority: **P0** is required to pass Phase 1, **P1** is strongly expected, **P2** is nice to have.

### 5.1 Data collection
| ID | Requirement | Priority |
|---|---|---|
| FR-C1 | Collect text from at least 5 domain categories: literature, news, education/science, culture/history, public information, web, oral | P0 |
| FR-C2 | Record every source in `metadata/sources.csv` using the fields of the LDC-IL metadata schema: file, title, author/editor, publisher, year, category, subcategory, source URL, format, licence, collected_by, date | P0 |
| FR-C3 | Keep raw files untouched in `data/raw/<category>/`. All transformations write to `data/interim/` or `data/processed/` | P0 |
| FR-C4 | Reach ≥ 2M clean tokens. Candidate sources: more *Bhaangarbhuin* e-paper issues, Konkani Wikipedia (Devanagari), AI4Bharat IndicCorp v2 / Sangraha Konkani subsets, Goa government / Directorate of Official Language documents, Konkani blogs and magazines, SCERT Goa textbooks | P0 |
| FR-C5 | Respect robots.txt and site terms when scraping, and record the licence or permission status | P1 |
| FR-C6 | Domain balance: no single category above 60% of final tokens | P2 |

### 5.2 Extraction and format conversion
| ID | Requirement | Priority |
|---|---|---|
| FR-E1 | PDF with a valid Unicode text layer → extract text (`pdftotext`/PyMuPDF) | P0 |
| FR-E2 | PDF without a usable layer (legacy fonts, scans) → render at ≥ 300 dpi → OCR (Tesseract `hin+mar`, or EasyOCR `mr`) | P0 |
| FR-E3 | HTML/XML → strip tags and boilerplate. LDC-IL XML → `scripts/extract_ldcil.py` ✅ | P0 |
| FR-E4 | DOCX/other formats → plain text | P1 |
| FR-E5 | Convert UTF-16 to UTF-8 (`iconv`) and CRLF to LF (`dos2unix`). Every output passes the `file` check | P0 |
| FR-E6 | Report OCR quality (CER/WER) against a hand-typed gold reference. Done for `GHANAGHAINIYATICHE` vs LDC-IL `KO00002` (`scripts/ocr_quality_check.py`) ✅. Extend to newspaper OCR using about 3 hand-typed pages | P1 |

### 5.3 Cleaning and normalisation
| ID | Requirement | Priority |
|---|---|---|
| FR-N1 | Unicode NFC normalisation; consistent nukta, chandrabindu/anusvara and ZWJ/ZWNJ handling | P0 |
| FR-N2 | Remove HTML remnants, page numbers, running headers, title pages, publisher/ISBN/© lines and English blurbs | P0 |
| FR-N3 | Line filter: drop lines with Devanagari character ratio < 0.7 or made up mostly of digits/Latin/other scripts (e.g. stray Tamil/Hangul OCR output) | P0 |
| FR-N4 | Fix OCR spacing: remove the space before punctuation, rejoin hyphenated and line-wrapped words into paragraphs | P0 |
| FR-N5 | Separate punctuation from tokens before counting. Handle numbers (Devanagari `०–९` and ASCII) with a `<NUM>` class in the analysis | P0 |
| FR-N6 | **Language identification**: score each paragraph as Konkani vs Marathi/Hindi using marker words (`आसा, आनी, हांव, म्हजो, जाल्यार` vs `आहे, आणि, मी, माझा, है, और`) and drop non-Konkani paragraphs | P0 |
| FR-N7 | Deduplication: exact (hash of normalised paragraphs) plus near-duplicate (line Jaccard / MinHash ≥ 0.8). Log every exclusion in `metadata/excluded_documents.csv` ✅ (initial entries done) | P0 |
| FR-N8 | Flag orthographically impossible tokens (vowel sign at word start, doubled matras, `U+FFFD`) | P1 |
| FR-N9 | Optional OCR post-correction, e.g. restoring dropped anusvara (the most frequent OCR error: 118 of the measured edits), using frequency list + edit distance + LM ranking | P2 |
| FR-N10 | **Iterative cleaning**: re-run analyses after each cleaning iteration and record before/after statistics in `docs/data_cleaning_log.md` | P0 |

### 5.4 Analysis
| ID | Requirement | Priority |
|---|---|---|
| FR-A1 | **Type–token analysis**: corpus size (tokens), vocabulary (types), TTR, hapax count/ratio | P0 |
| FR-A2 | Frequency of every type, sorted in decreasing order → `results/word_frequencies.csv` (rank, type, freq, rel_freq, cum_coverage) | P0 |
| FR-A3 | **Coverage analysis**: number of types needed for 10, 25, 50, 75, 90, 95, 99% token coverage → `results/coverage_analysis.csv` + curve | P0 |
| FR-A4 | Analyses for **iteration 0 (raw)** and each cleaned iteration, overall and per category | P0 |
| FR-A5 | Zipf rank–frequency (log-log) and Heaps' law (vocabulary growth) plots | P1 |
| FR-A6 | Compare our type growth with LDC-IL's statement on Konkani "saturation" | P2 |

### 5.5 Sentence segmentation
| ID | Requirement | Priority |
|---|---|---|
| FR-S1 | Split paragraphs into sentences on `।`, `॥`, `?`, `!` and `.` | P0 |
| FR-S2 | Do not split on `.` inside abbreviations/initials (`के.के.`, `पी.ओ.`, `डॉ.`, `आर्.पी.आय.`), decimals (`3.5`) or ellipses; handle quoted speech | P0 |
| FR-S3 | Output one sentence per line (`sentences.txt`) plus `sentences.tsv` (doc_id, sent_id, n_tokens, sentence) | P0 |
| FR-S4 | Handle verse separately (`॥ १ ॥` verse markers) | P1 |
| FR-S5 | Evaluate on ~200 hand-annotated paragraphs, including the LDC-IL gold sample, and report boundary precision/recall | P1 |

### 5.6 Length analyses
| ID | Requirement | Priority |
|---|---|---|
| FR-L1 | Sentence-length distribution (tokens): histogram, mean, median, std, percentiles, overall and per category | P0 |
| FR-L2 | Word-length distribution in **characters (code points)** and **aksharas (grapheme clusters)**, type-weighted and token-weighted | P0 |
| FR-L3 | Plots saved to `results/plots/` with labelled axes and captions | P0 |

### 5.7 Reporting
| ID | Requirement | Priority |
|---|---|---|
| FR-R1 | `docs/report.md` (and PDF): sources → extraction → cleaning iterations with before/after tables → statistics → coverage → segmentation → length analyses → **observations and inferences** → limitations → readiness for Phase 2 | P0 |
| FR-R2 | Every number in the report is reproducible from a committed script | P0 |
| FR-R3 | `metadata/data_dictionary.md` documents every output file and column | P1 |

### 5.8 Phase 2 hand-off
| ID | Requirement | Priority |
|---|---|---|
| FR-H1 | Freeze **corpus v1.0** with a tag and a release note | P0 |
| FR-H2 | Fixed **document-level** train/dev/test split (80/10/10) in `data/splits/`, so no sentences from one book appear in two splits | P0 |

---

## 6. Scope and requirements: Phase 2 (Representations and LMs)

> The official Phase 2/3 brief is not out yet. The instructor said only that later phases cover "word embeddings, Language Models, Neural Networks, Named Entities etc." The requirements below are our proposal and will be revised when the brief arrives.

| ID | Requirement | Priority |
|---|---|---|
| FR-P2-1 | Tokenisers: whitespace+punctuation baseline and SentencePiece BPE/Unigram (8k–32k vocab), reporting fertility and OOV rate | P0 |
| FR-P2-2 | N-gram LMs (uni/bi/tri, plus 4/5-gram via KenLM) with Laplace, Good-Turing and Kneser-Ney smoothing; perplexity on the test split | P0 |
| FR-P2-3 | Word embeddings: word2vec (CBOW, skip-gram), FastText (subword), GloVe | P0 |
| FR-P2-4 | Konkani intrinsic evaluation set: about 200 word-similarity pairs and about 100 analogies, built by the team | P1 |
| FR-P2-5 | Nearest-neighbour inspection and t-SNE/UMAP visualisations | P1 |
| FR-P2-6 | Compare with pretrained multilingual resources (IndicBERT/MuRIL tokenisers, FastText `gom`) | P2 |
| FR-P2-7 | Learning curve: perplexity and similarity scores with 10/25/50/100% of the corpus | P1 |

## 7. Scope and requirements: Phase 3 (Neural models and applications)

| ID | Requirement | Priority |
|---|---|---|
| FR-P3-1 | Neural LMs: feed-forward (Bengio-style) → LSTM/GRU → small Transformer, compared by perplexity with the Phase 2 n-gram LMs | P0 |
| FR-P3-2 | **NER dataset**: 2–3k sentences annotated with PER, LOC, ORG, MISC, DATE in BIO format using Doccano/Label Studio, with written guidelines | P0 |
| FR-P3-3 | Inter-annotator agreement (Cohen's κ) on a ≥ 200-sentence shared subset | P0 |
| FR-P3-4 | NER models: CRF baseline → BiLSTM-CRF with our FastText → fine-tuned MuRIL/IndicBERT; entity-level P/R/F1 (seqeval) | P0 |
| FR-P3-5 | Continued MLM pretraining of a multilingual model on our corpus | P2 |
| FR-P3-6 | Optional: POS tagging (BIS tagset), domain text classification (our categories as labels), OCR spell-corrector | P2 |

---

## 8. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 **Encoding** | All text artefacts are UTF-8, LF line endings, NFC. Enforce with a `.gitattributes` rule `* text=auto eol=lf` and a check in `validate_konkani.py` |
| NFR-2 **Reproducibility** | Each stage is one script with CLI arguments and fixed random seeds; the pipeline re-runs from `data/raw` end to end |
| NFR-3 **Traceability** | Every output line traces back to a `doc_id` → `sources.csv` |
| NFR-4 **Performance** | The full Phase 1 pipeline (excluding OCR) runs in under 30 minutes on a team laptop; OCR is batch-run once and cached in `data/interim/` |
| NFR-5 **Portability** | Runs on Windows and Linux with Python ≥ 3.10; dependencies pinned in `requirements.txt` |
| NFR-6 **Code quality** | Scripts are small and single-purpose, with a usage docstring; no hard-coded absolute paths |
| NFR-7 **Legal/ethical** | Copyrighted texts are used for non-commercial academic research only; licences are recorded; repo visibility is reviewed before release |
| NFR-8 **Versioning** | Corpus releases are tagged (`corpus-v1.0`, …); large derived files are kept out of git if they exceed GitHub limits (use releases or Git LFS) |

## 9. System overview

```
data/raw/<category>/            (PDF, OCR txt, XML, HTML)
        │  extract_text.py / extract_ldcil.py   [+ OCR]
        ▼
data/interim/<category>/*.txt   (plain UTF-8, LF)
        │  clean_text.py → validate_konkani.py → deduplicate.py
        ▼
data/processed/by_category/<cat>/corpus_clean.txt
        │  segment_sentences.py
        ▼
data/processed/combined/{konkani_devanagari_clean.txt, sentences.txt, sentences.tsv, documents.jsonl}
        │  analyze_corpus.py                    ocr_quality_check.py (gold vs OCR)
        ▼                                       ▼
results/{corpus_statistics.txt, word_frequencies.csv, coverage_analysis.csv, plots/}   results/ocr_quality/
        │
        ▼
data/splits/{train,dev,test}.txt  → Phase 2 (tokenisers, LMs, embeddings) → Phase 3 (neural LMs, NER)
```

## 10. Success metrics and acceptance criteria

### Phase 1 (exit criteria)
| Metric | Target | Current (25 Oct 2026) |
|---|---|---|
| Clean tokens | ≥ 2M (stretch 5M) | ~0.55M raw, not yet cleaned |
| Categories with data | ≥ 5 | 4 (literature, culture/history, partial education, LDC-IL sample); news needs OCR |
| Files passing `file` = "UTF-8 text", no CRLF | 100% | 0% of raw `.txt` (all CRLF); LDC-IL interim files ✅ |
| Non-Devanagari characters in final corpus (excluding punctuation/digits) | < 0.5% | 0.1–3.3% per raw file |
| Estimated non-Konkani (Marathi/Hindi) paragraphs remaining | < 2% on a 200-paragraph manual audit | not measured |
| Exact duplicate paragraphs | 0 | 1 duplicate book found and logged |
| OCR CER on gold reference | Reported per OCR source | 4.77% (literature OCR) ✅ |
| Sentence segmentation F1 | ≥ 0.95 on the annotated sample | not built |
| Report | Covers all 10 brief items, with observations and inferences per step | not started |

### Phase 2
- Kneser-Ney trigram perplexity is lower than the Laplace trigram on the same test split, with a clear table.
- FastText beats word2vec on OOV and misspelt-word similarity, as checked on the team's evaluation set.

### Phase 3
- The neural LM's perplexity is lower than the best n-gram LM.
- NER dataset with κ ≥ 0.7; the best model's entity F1 is reported alongside the CRF baseline.

## 11. Milestones

| Milestone | Contents | Target |
|---|---|---|
| M1 Audit + plan | Repo audit, README, PRD, LDC-IL integration, OCR CER | ✅ Week 0 |
| M2 Data ≥ 2M raw tokens | OCR of newspapers and textbook, web/Wikipedia/IndicCorp collection, `sources.csv` complete | Week 1–2 |
| M3 Iteration-0 analysis | Raw type/token, frequency, coverage + observations | Week 2 |
| M4 Clean corpus | Cleaning, language ID, dedup; iterations 1…n with before/after stats | Week 2–3 |
| M5 Segmentation + lengths | Sentences, evaluation, length plots | Week 3 |
| M6 Phase 1 submission | Report, corpus v1.0 tag, splits | Week 4 |
| M7 Phase 2 | Tokenisers, LMs, embeddings, evaluation | per course schedule |
| M8 Phase 3 | Neural LMs, NER dataset and models, final report | per course schedule |

**Ownership:** Member 1 handles the pipeline, Member 2 data collection, Member 3 analysis and plots, Member 4 linguistic QA and the report. Everyone annotates in Phase 3.

## 12. Risks and mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Not enough Konkani text to reach 2M tokens | Weak Phase 2/3 results | Medium | Prioritise newspaper OCR and web/IndicCorp sources early; track tokens weekly |
| No Konkani OCR model; legacy-font PDFs | Noisy news data | High | Use Tesseract `hin+mar`; measure CER on hand-typed pages; post-correct the common errors (anusvara, virama) |
| Marathi/Hindi contamination | Wrong statistics and embeddings | Medium | Marker-word language ID plus a manual audit sample |
| Full LDC-IL corpus unavailable | Less clean gold data | Certain | Use the sample as a gold reference and for methodology only |
| Dialect and spelling variation (Antruzi, Saxtti, Mangalorean, Kerala Konkani) | Inflated types | High | Document it; normalise only where `KONKANI SHUDDLEKHANACHE NEM` rules are clear; report the effect |
| Copyright of books and e-papers | Cannot publish data openly | Medium | Academic use only; consider making the repo private; publish statistics and models instead of raw text |
| Large files exceed GitHub limits | Push failures | Medium | Git LFS or GitHub Releases for the corpus; keep only scripts and small samples in git |
| CRLF reintroduced by Windows git checkouts | Fails the `file`/LF requirement | Medium | `.gitattributes` with `eol=lf` and a validation script check |
| Annotation effort for NER underestimated | Phase 3 delays | Medium | Start the guidelines early, pre-annotate with gazetteers, split work across 4 annotators |

## 13. Assumptions and dependencies
- The team has Python 3.10+, `pdftotext`, PyMuPDF and rapidfuzz; Tesseract with `hin`/`mar` traineddata will be installed for OCR.
- The instructor accepts OCR-derived text provided its quality is measured and reported.
- Phase 2/3 GPU needs are modest (a small Transformer, MuRIL fine-tuning) and can be met with Colab or Kaggle.

## 14. Open questions
1. Does the instructor require a minimum corpus size for Phase 1 grading?
2. Can openly licensed third-party corpora (IndicCorp, Wikipedia) be counted towards our collected corpus, or must data be collected by the team?
3. Should the final corpus keep or drop poetry/verse? LDC-IL drops verse found inside prose.
4. Which Phase 2/3 tasks are mandatory once the official brief is out?
5. Should the repository stay public, given the copyrighted sources?

## 15. Glossary
| Term | Meaning |
|---|---|
| Token | One occurrence of a word form in running text; the total count is the corpus size |
| Type | A unique word form |
| TTR | Type–token ratio |
| Hapax legomenon | A type that occurs exactly once |
| Coverage | Fraction of tokens accounted for by the top-N most frequent types |
| Akshara | An orthographic syllable / grapheme cluster in Devanagari, e.g. `क्ष` |
| CER / WER | Character / word error rate: edit distance divided by reference length |
| NFC | Unicode canonical composition normal form |
| LDC-IL | Linguistic Data Consortium for Indian Languages, CIIL Mysore |
| Perplexity | Exponential of the average negative log-likelihood per token; lower is better |
| κ (kappa) | Cohen's kappa, chance-corrected inter-annotator agreement |
