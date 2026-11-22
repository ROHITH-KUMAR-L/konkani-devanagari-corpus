# Progress log: 20–22 Nov 2026 (Fedora work sessions)

What changed from the move to the Fedora machine (20 Nov 2026) to 22 Nov 2026, grouped by stage. Commits are on `main`; git history has the detail.

Before these sessions the repo had: README, PRD, the LDC-IL gold sample and OCR quality check, the e-paper fetch + OCR script, and 21 OCR'd Bhaangarbhuin issues (360k tokens). No cleaning pipeline, analyses, splits or Phase 2/3 work existed.

---

## Phase 1: corpus building and analysis

### Stage 1.1: Collection (brief steps 1–3)
- Linux support for `fetch_epaper.py` (system tessdata auto-detected); new `--force-ocr` flag.
- Textbook `fkokb1ps.pdf` re-OCR'd at 300 dpi (its PDF text layer has broken vowel signs).
- Konkani Wikipedia dump (gomwiki, CC BY-SA) added as the 5th domain, `06_web_articles`.
- Bhaangarbhuin e-paper fetch ran to the paywall: **573 issues** (2 Apr 2025 – 20 Nov 2026), up from 21. OCR text per issue is in `data/interim/02_news_and_media/epaper/BHAANGARBHUIN/`; one row per issue in `metadata/epaper_issues.csv`. The title list shows no other Konkani title on the site (the others are English or Marathi).
- Category 05 (Goa government press releases) tried and dropped: the DIP site is down, `goa.gov.in` blocks automated access, and the releases are in English.
- `metadata/sources.csv` filled from book front matter.

### Stage 1.2: Extraction (iteration 0)
- `scripts/extract_text.py`: every source to a document tree `data/interim/iter0/<category>/<doc_id>.txt`, plus manifest `metadata/documents.csv`.
- `scripts/corpus_utils.py`: shared tokeniser (numbers to `<NUM>`), akshara splitter and Devanagari ratio.

### Stage 1.3: Cleaning iterations 1–3
- Iteration 1 `clean_text.py`: NFC, junk lines, running headers, boilerplate, other scripts, punctuation spacing, paragraph rejoin. Log: `metadata/clean_log.csv`.
- Iteration 2 `validate_konkani.py`: Konkani vs Marathi/Hindi language ID, orthography and numeric-table filters. Log: `metadata/quality_checks.csv`.
- Iteration 3 `deduplicate.py`: exact paragraph hash + MinHash near-duplicate documents. Log: `metadata/dedup_log.csv`.
- Anusvara post-correction deliberately skipped: both spellings (e.g. आसा / आसां) are real words.

### Stage 1.4: Analysis (brief steps 4–7)
- `analyze_corpus.py`: tokens, types, TTR, hapax, frequency list, coverage, Zipf and Heaps fits, plots; overall and per category, for every iteration (`results/iter0..2`, final in `results/`).
- `document_audit.py`: tokens kept per document per iteration (`results/document_audit.csv`).
- `length_analysis.py`: sentence and word length (characters and aksharas) with plots.

### Stage 1.5: Sentence segmentation and evaluation
- `segment_sentences.py`: rules for abbreviations, initials (Devanagari and English letter names), decimals, list numbers (also without a space, e.g. `2.त्या`), datelines, verse markers.
- Gold sets (first pass by Claude, **team must verify**): `data/eval/segmentation_gold.txt` (40 dev paragraphs), `segmentation_gold_heldout.txt` (154 held-out paragraphs). Held-out F1 **0.990** after an independent AI second-pass review (`data/eval/review/`).
- Language audit: `data/eval/language_audit_sample.tsv`, 200 paragraphs, 1.0% non-Konkani (target < 2%).

### Stage 1.6: Splits, report, automation
- `make_splits.py`: document-level 80/10/10 split (seed 2026) to `data/splits/`, plus `data/splits/balanced/` where no category exceeds 60% of tokens.
- `build_report.py`: `docs/report.md` and `docs/data_cleaning_log.md` are generated from `docs/templates/`. Edit the templates, not the outputs.
- `scripts/run_phase1.sh`: the whole Phase 1 chain in one command.
- `scripts/rebuild_all.sh`: Phase 1 + Phase 2 + teammate zip in one command (optionally waits for the fetch to end).
- `metadata/data_dictionary.md`, `docs/collection_guidelines.md` updated.

### Phase 1 numbers (final run, 22 Nov 2026, all 573 e-paper issues; tag `corpus-v1.0`)

| | Value |
|---|---|
| Clean tokens | **10.51M** (goal ≥ 2M, stretch 5M); 541,967 types |
| Documents | 613, 5 domains |
| Share by domain | news 86.0%, web 10.3%, literature 2.2%, culture 1.5%, education 0.05% |
| Splits (tokens) | train 8.46M, dev 1.01M, test 1.04M (490/61/61 documents) |
| Balanced subset | 3.66M tokens, news 59.7% |

Held-out segmentation F1 0.990. The previous run (289 issues) gave 6.06M tokens. Phase 2 and 3 were re-run on the final splits on 22 Nov; the numbers below are final-corpus numbers.

---

## Phase 2: tokenisers, n-gram LMs, embeddings

### Stage 2.1: Tokenisers and n-gram LMs
- Code: `scripts/phase2/{common,tokenizers,ngram_lm}.py`, run with `scripts/run_phase2_lm.sh` (Python 3.12 venv `.venv-phase2`, because gensim does not build on 3.14).
- Tokenisers: whitespace baseline and SentencePiece BPE/Unigram at 8k–32k; BPE-32k has the lowest fertility (1.29 pieces per word, earlier splits).
- Word n-gram LMs, 1–4-gram, with Laplace, add-k, Katz (Good-Turing) and Kneser-Ney / modified KN. Best: **modified-KN 4-gram, test perplexity 484**.
- Subword n-gram LMs over SentencePiece pieces up to 5-gram: per-word perplexity 1,121. Unseen words in test: 4.1% of tokens with whole words, 0 with subwords.
- Learning curve at 10/25/50/100% of the training data.
- Learning curve: trigram perplexity 1,400 → 932 → 679 → 503 at 10/25/50/100% of the data (fixed vocabulary); not saturated.
- Write-up: [docs/phase2_lm.md](phase2_lm.md).

### Stage 2.2: Word embeddings
- Code: `scripts/phase2/{train_embeddings,eval_embeddings}.py`, run with `scripts/run_phase2_embeddings.sh` (about 23 min).
- Models: word2vec CBOW and skip-gram, FastText, GloVe.
- Test sets (draft by Claude, **team must verify**): `data/eval/word_similarity.tsv` (213 pairs), `data/eval/analogies.tsv` (118 analogies).
- Results: Spearman ρ CBOW 0.350, skip-gram 0.439, **FastText 0.466** (2–5-character pieces 0.447), GloVe 0.236. Analogies best for FastText with 2–5-character pieces (50 of 118). FastText keeps OCR-damaged words close to the original (cosine 0.73).
- Write-up: [docs/phase2_embeddings.md](phase2_embeddings.md).
- Nearest neighbours, t-SNE plots, OOV robustness and a learning curve in `results/phase2/`.

---

## Phase 3: neural LMs and NER

### Stage 3.1: Neural language models
- Code: `scripts/phase3/neural_lm.py`, run with `scripts/run_phase3_lm.sh` (PyTorch CPU; installs torch into `.venv-phase2` if missing).
- Models: feed-forward (Bengio), LSTM, small Transformer, character LSTM; same splits and same per-word normalisation as the n-gram models.
- Test per-word perplexity: LSTM trained 25 min **1,389**, Transformer 25 min 1,587, against the best subword KN 5-gram 1,121. The neural models are compute-bound on a CPU (dev perplexity still falling); KN profits from the larger corpus immediately.
- Write-up: [docs/phase3_neural_lm.md](phase3_neural_lm.md).

### Stage 3.2: Named entity recognition
- Guidelines: [docs/ner_guidelines.md](ner_guidelines.md) (PER, LOC, ORG, MISC, DATE; BIO; Konkani case suffixes, honorifics, transliterated names).
- `data/eval/ner/pool.conll`: 3,000-sentence stratified pool (seed 2026). Do not regenerate it once annotation starts.
- Silver annotation of 300 sentences (AI first pass, **team must verify**): `data/eval/ner/silver_{all,dev,test}.conll`, 350 entities.
- Baselines (`scripts/phase3/ner_baselines.py`): gazetteer + rules F1 0.494, CRF F1 0.394 on the silver test half. PER recall is very low. Write-up: [docs/phase3_ner.md](phase3_ner.md).

---

## Still to do

1. Train the neural LMs longer (or on a GPU) to pass the KN baseline.
2. Team verification of every AI-drafted set: segmentation gold, language audit, word similarity, analogies, NER silver (see each doc).
3. NER: grow to 2–3k verified sentences, 200 double-annotated for Cohen's κ; BiLSTM-CRF and MuRIL/IndicBERT.
4. Hand-typed gold for about 3 e-paper pages (newspaper OCR CER).
5. Final report and presentation.
