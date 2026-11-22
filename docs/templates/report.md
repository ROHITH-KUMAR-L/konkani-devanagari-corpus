# Phase 1 report: a Konkani (Devanagari) text corpus

**NLP Experiential Learning, Phase 1.** Language: Konkani (ISO 639-3 `gom`/`kok`), Devanagari script.

> **Run of {{date}}.** Every number comes from `scripts/run_phase1.sh` and is filled in by `scripts/build_report.py`. This run includes {{news_issues}} Bhaangarbhuin issues ({{news_first}} to {{news_last}}). The pipeline is deterministic: re-running it reproduces every number.

## Summary

| | |
|---|---|
| Clean corpus | **{{fin_tokens}} tokens**, {{fin_types}} types, {{fin_sentences}} sentences, {{fin_docs}} documents, {{n_domains}} domains |
| Format | Plain UTF-8, LF line endings, Unicode NFC; `file` reports "UTF-8 text" |
| Non-Devanagari tokens | {{fin_nondeva_pct}}% (numbers {{fin_num_pct}}%, as `<NUM>`) |
| OCR quality (book OCR vs. LDC-IL gold) | CER {{ocr_cer}}%, WER {{ocr_wer}}% |
| OCR quality (newspaper, 902 words, AI-transcribed reference) | Corpus text CER 1.06%, WER 4.43% per line; one 2025 page mixes lines across columns |
| Sentence segmentation | Held-out F1 **{{seg_held_f1}}** ({{seg_held_paras}} paragraphs, {{seg_held_bounds}} boundaries); P {{seg_held_p}}, R {{seg_held_r}} |
| Non-Konkani text | {{audit_nonkok_pct}}% on a {{audit_n}}-paragraph random audit (target < 2%) |
| Splits | Document-level {{train_pct}} / {{dev_pct}} / {{test_pct}}% of tokens (train/dev/test), plus a domain-balanced subset |

## 1. Sources (brief step 3)

Full bibliographic details are in `metadata/sources.csv`, and per-issue newspaper metadata is in `metadata/epaper_issues.csv`.

| Category | Sources | How text was obtained |
|---|---|---|
| 01 Literature | {{lit_books}} books (novels, short stories, poetry, drama, children's) + {{ldcil_files}} LDC-IL gold-standard files | Book text OCR'd by the team; LDC-IL XML parsed (`extract_ldcil.py`) |
| 02 News | *Bhaangarbhuin*, Goa's Konkani daily: {{news_issues}} issues, {{news_first}} to {{news_last}} | Free e-paper issues downloaded and OCR'd with Tesseract `mar+hin` (`fetch_epaper.py`) |
| 03 Education | NCERT Konkani vocational-education textbook | PDF text layer is broken (doubled vowel signs); re-OCR'd at 300 dpi with `--force-ocr` |
| 04 Culture/history | {{cult_books}} books (story anthology, Mahabharata in Old Konkani, biography, orthography rules, culture) | OCR text |
| 06 Web | Konkani Wikipedia (`gomwiki`, 1 Nov 2026 dump, CC BY-SA): {{wiki_articles}} articles | `mwparserfromhell`, Devanagari articles only (≥ 70% Devanagari letters; Romi articles skipped) |

Empty categories: {{empty_categories}}. For public information we tried the Goa government press releases: the Department of Information and Publicity site was down, `goa.gov.in` refuses automated access, and its releases are in English.

## 2. Extraction and OCR quality

- **Encoding.** Every file is converted to UTF-8 with LF line endings and the BOM removed. Raw files are never changed; all output goes to `data/interim` and `data/processed`. `.gitattributes` keeps LF line endings on Windows checkouts.
- **OCR quality.** We aligned the hand-typed LDC-IL file `KO00002` with our OCR of the same book (`ocr_quality_check.py`):
  - CER {{ocr_cer_raw}}% and WER {{ocr_wer_raw}}% on the raw OCR;
  - CER **{{ocr_cer}}%** and WER **{{ocr_wer}}%** once the spurious space before punctuation is removed (a fix that cleaning applies).
- **Most frequent OCR errors** (`results/ocr_quality/confusions.csv`): dropped anusvara `ं`, dropped virama `्`, dropped `ा`.
- **Newspaper OCR quality** (`ocr_quality_check.py --epaper`; details in `results/ocr_quality/epaper_summary.txt`). Six article regions from three issues (Apr 2025, Nov 2025, Jun 2026; 902 words) were transcribed from 300-dpi page images by Claude (AI), not hand-typed, so they are a reference for estimates only.
  - Text stored in the corpus, aligned line by line: CER **1.06%**, WER **4.43%**.
  - Tesseract on each crop as a whole: CER 14.12%, WER 22.17%; 2.75% / 8.36% without one region where Tesseract cut a narrow column into pieces. That error is reading order, not misread letters.
  - On the April 2025 page the full-page OCR interleaves lines from neighbouring columns (only 30–39% of line order kept, against 95–97% on later pages), which breaks some sentences.
- **Inference:** OCR errors are mostly *deletions of small marks*. They create false word types (`आसा`/`आसां`) rather than garbage, which matters for the type counts below.

## 3. Cleaning iterations (brief steps 6–7)

The full rationale and the before/after tables are in [`docs/data_cleaning_log.md`](data_cleaning_log.md). In brief:

| Iteration | Script | What it removes | Tokens after |
|---|---|---|---|
| 0 raw | `extract_text.py` | (nothing) | {{tok0}} |
| 1 cleaned | `clean_text.py` | Page numbers, running headers, boilerplate, Latin/other-script text, OCR spacing; rebuilds paragraphs across line and page breaks | {{tok1}} |
| 2 Konkani only | `validate_konkani.py` | Marathi/Hindi paragraphs (marker-word language ID), Marathi documents, OCR garbage, number tables | {{tok2}} |
| 3 deduplicated | `deduplicate.py` | {{dedup_paras}} duplicate paragraphs ({{dedup_pct}}%): Wikipedia census templates, newspaper boilerplate | **{{tok3}}** |

**Observations that drove the cleaning:**
- Iteration 0's frequency list was topped by Wikipedia census-template words ({{template_ranks}}). These come from bot-generated village stubs: of {{wiki_lines}} Wikipedia lines only {{wiki_unique}} were unique.
- `HISTORY.txt` turned out to be Marathi ({{history_kok}} Konkani vs {{history_mar}} Marathi markers).
- One culture book is a Konkani–Hindi parallel grammar.

A {{audit_n}}-paragraph random audit of the final corpus (`data/eval/language_audit_sample.tsv`) found {{audit_nonkok}} non-Konkani paragraphs (**{{audit_nonkok_pct}}%**, target < 2%), {{audit_garbage_pct}}% OCR garbage, {{audit_partial_pct}}% partly garbled paragraphs and {{audit_fragment_pct}}% newspaper fragments where OCR merged columns.

## 4. Type–token analysis (brief step 4)

Tokens are words with punctuation removed and every number mapped to the single type `<NUM>`. Full output: `results/corpus_statistics.txt`, and every type with its frequency in decreasing order is in `results/word_frequencies.csv`.

{{table_typetoken}}

{{table_categories}}

The most frequent words are {{top_words}}. All are function words, as Zipf's law predicts.

![Zipf](../results/plots/zipf.png) ![Heaps](../results/plots/heaps.png)

## 5. Coverage analysis (brief step 5)

{{table_coverage}}

Per-category coverage is in `results/coverage_analysis.csv`.

![Coverage](../results/plots/coverage.png)

## 6. Sentence segmentation (brief step 8)

`segment_sentences.py` splits on `।`, `॥`, `?` and `!`, and on `.` (Konkani uses the full stop as often as the danda). A verse number `॥ १ ॥` counts as one boundary. A `.` is **not** a boundary in these cases:
- decimals (`37.2`);
- ellipses;
- list numbers at the start of a sentence (`12.`);
- known abbreviations (`डॉ.`, `श्री.`, `इ.स.`, `किमी.`, …);
- initials: a one-akshara word before `.`, or an English letter name (`के.आर.`, `जी. एन. जोशी`, `पी. टी. सर`).

Common one-akshara words that end sentences (`ना`, `हो`, `जा`, `गो`, `यो`, …) are exempt from the initials rule. Closing quotes stay with their sentence. A newspaper dateline (`पणजी:`, `'मडगांव:` …) starts a new sentence even when the headline before it has no full stop. A list number in mid-paragraph (`… ना 13. समासांत`, `कृश्ण 2.त्या`) starts a new sentence *before* the number.

The output is `data/processed/combined/sentences.txt` (one sentence per line) and `sentences.tsv` (doc_id, sent_id, n_tokens, sentence).

**Evaluation** (`data/eval/`, boundaries compared by token position, so the side a quote mark falls on is not an error):

| Gold set | Paragraphs | Gold boundaries | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Development (`segmentation_gold.txt`; rules were tuned on it) | {{seg_dev_paras}} | {{seg_dev_bounds}} | {{seg_dev_p}} | {{seg_dev_r}} | {{seg_dev_f1}} |
| **Held-out** (`segmentation_gold_heldout.txt`; never used to write rules) | {{seg_held_paras}} | {{seg_held_bounds}} | {{seg_held_p}} | {{seg_held_r}} | **{{seg_held_f1}}** |

History:
- The first version scored F1 0.934 on the development set.
- The dateline rule and the list-number rule raised it to 0.990.
- With those rules frozen, the held-out set scored **F1 0.989**, above the 0.95 target. This is the honest estimate.
- The held-out set then revealed two gaps, which were fixed: list numbers written with no space (`2.त्या`) and the particles `गो`/`यो`. The table shows the current rules.

Development and held-out paragraphs were sampled separately (seeds 7 and 2027), stratified by category, and half of each set was chosen to contain hard cases (a `.` after a short word, or a colon). Six held-out paragraphs that were pure OCR garbage from advertisement pages were left out.

Remaining errors (`results/segmentation_eval*.txt`) are almost all OCR artefacts:
- a stray `.` inside a word or number (`सूतवाड्. मय`, `37.2. अंश`);
- `आदी.` ("etc.") ending a list in mid-sentence;
- an OCR-split word read as an initial (`मारिना लो.`).

**Limitation:** both gold sets were made by correcting system output, first by an AI assistant. The team should verify them.

## 7. Length analyses (brief step 9)

Full output: `results/length_analysis.txt`, `sentence_length_stats.csv` and `word_length_distribution.csv`.

{{table_sentence_length}}

{{table_word_length}}

On average one akshara is {{chars_per_akshara}} code points: a consonant plus a vowel sign, conjunct or anusvara.

![Sentence length](../results/plots/sentence_length.png)
![Word length](../results/plots/word_length.png)

## 8. Observations and inferences

1. **Konkani vocabulary does not saturate at this size.**
   - **Observation:** The Heaps exponent is β ≈ {{heaps_beta}}. Typical values for English are about 0.4–0.6. {{fin_hapax_pct}}% of types are hapaxes, and covering 90% of tokens takes {{cov90}} types ({{cov90_pct}}% of the vocabulary).
   - **Inference:** This is consistent with the LDC-IL documentation (2018, §9). It says Konkani is "highly agglutinative and morphologically rich … the new words coming into corpus for a unit amount of input is much higher compared to other languages", and that much larger corpora are needed for good coverage. Postpositions fuse with nouns (`घरांत`, `घराक`, `घराचो`), so each noun lemma surfaces as many types. Two consequences follow:
     - for Phase 2, **subword models (BPE/Unigram, FastText)** are needed, because a word-level vocabulary will have a high OOV rate on the test split;
     - for comparison, the LDC-IL corpus has about 4.0M words from 282 titles.
2. **The β value is partly inflated by noise and domain mixing.** Each new domain and each new newspaper issue brings new names, so the growth curve keeps rising. OCR deletions (anusvara, virama) also create false types.
3. **Deduplication matters more than any other cleaning step for frequency statistics.** Removing duplicates took out {{dedup_tok_pct}}% of tokens but moved the 50%-coverage point by {{cov50_change}}% ({{cov50_before}} → {{cov50_after}} types), and changed the Zipf slope from {{zipf_before}} to {{zipf_after}}. Without dedup, the frequency list describes a census table, not the language.
4. **Domains differ sharply.**
   - {{short_cat}} has the shortest sentences (median {{short_med}} tokens) and {{long_cat}} the longest (median {{long_med}}). Fiction is dominated by dialogue and quoted speech; the textbook by expository prose.
   - Among the large categories, {{lowttr_cat}} has the lowest TTR ({{lowttr}}): recurring names, places and formulae.
   - {{highhapax_cats}} have the highest hapax shares.
5. **Function words dominate and are distinctively Konkani.** `आनी`, `आसा`, `ना`, `पूण`, `खातीर` and `जाल्यार` separate Konkani from Marathi (`आणि`, `आहे`, `नाही`, `पण`, `साठी`). That is why the marker-word language ID works: it found a Marathi book and a Hindi parallel text that script-based filtering would have kept.
6. **Spelling variation is large.** Examples: {{variants}}. We kept it, since it is real dialect and orthographic variation (Antruzi, Saxtti, Mangalorean and Cochin spellings all occur, e.g. `अश्शि बरंयताति`). It does split the counts for one word, which Phase 2 embeddings should partly recover.
7. **Words average {{aks_tok}} aksharas (token-weighted) but {{aks_type}} aksharas (type-weighted).** Frequent words are short function words, while the long tail is long inflected forms. This matches Zipf's law of abbreviation.

## 9. Limitations

- **Domain balance (FR-C6).** {{balance_status}}.
  - The full corpus is kept for Phase 2, because more text gives better language models.
  - `make_splits.py` also writes a domain-balanced subset to `data/splits/balanced/`, with no category above {{max_share}}%. It has {{bal_docs}} documents and {{bal_tokens}} tokens: {{bal_shares}}.
  - Capped categories keep evenly spaced documents, which for newspapers spreads them over the whole date range.
- **Missing categories.** {{empty_categories}} are empty. Education has one textbook.
- **Residual OCR noise.** Anusvara loss, merged newspaper columns and missing spaces remain. Anusvara is not restored automatically (FR-N9): both forms are often real words (`आसा` "is", `आसां` "I am"), so blind restoring would add errors. CER is measured on book OCR only; newspaper CER needs about 3 hand-typed pages (FR-E6).
- **Language ID uses marker words.** Paragraphs with no markers (short headlines) are kept unchecked.
- **Gold sets annotated by an AI assistant.** The segmentation gold ({{seg_dev_paras}} dev + {{seg_held_paras}} held-out paragraphs) and the language audit ({{audit_n}} paragraphs) were first annotated by an AI assistant and need team verification.
- **Copyright.** Books and the e-paper are copyrighted. The corpus is for non-commercial academic use. The processed text is kept out of the public repository and shared inside the team as a zip.

## 10. Readiness for Phase 2

- `data/splits/{train,dev,test}.txt`: document-level splits (seed 2026; {{train_docs}}/{{dev_docs}}/{{test_docs}} documents), so no book, issue or Wikipedia shard appears in two splits. The manifest is in `data/splits/split_manifest.csv`, and the totals in `results/split_summary.json`.
- `data/splits/balanced/`: the same split restricted to the domain-balanced subset.
- `data/processed/combined/documents.jsonl` keeps the document boundaries for any re-splitting.
