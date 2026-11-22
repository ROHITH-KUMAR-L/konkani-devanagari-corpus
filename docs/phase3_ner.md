# Phase 3: Named Entity Recognition (setup and first baselines)

Status: **setup only.** The labels used here are an AI first-pass *silver* annotation of 300 sentences. They have not been checked by a human, so every number below is provisional. The numbers become reportable only after the team verifies the silver set (section 5).

## 1. What was built

| File | Purpose |
|---|---|
| `docs/ner_guidelines.md` | Annotation guidelines: PER/LOC/ORG/MISC/DATE, BIO, Konkani suffix, honorific and Goa place-name rules |
| `scripts/phase3/sample_ner_sentences.py` | Stratified sample (seed 2026) from `data/processed/combined/sentences.tsv` into `data/eval/ner/pool.conll` |
| `scripts/phase3/build_ner_silver.py` | Converts `silver_spans.txt` to CoNLL and splits it 50/50 per category into dev and test |
| `scripts/phase3/ner_common.py` | CoNLL I/O, BIO span extraction, case-suffix stripping, gazetteer matching |
| `scripts/phase3/ner_baselines.py` | Rule/gazetteer baseline and CRF (sklearn-crfsuite); seqeval entity-level P/R/F1 |
| `data/eval/ner/pool.conll` | 3,000 sentences to annotate, all tags `O` |
| `data/eval/ner/silver_spans.txt` | Silver annotation source: one line per sentence, spans as `TYPE:tokens` |
| `data/eval/ner/silver_{all,dev,test}.conll` | Silver CoNLL: 300 sentences, split 151 dev / 149 test |
| `data/eval/ner/gazetteer_seed.tsv` | Hand-written seed gazetteer (Goa, India and world places; parties and institutions; languages, religions and festivals) |
| `results/phase3/ner_label_counts.csv` | Entity counts per split |
| `results/phase3/ner_scores.csv` | Per-type and micro/macro P/R/F1 on silver_test |
| `results/phase3/ner_cv.csv` | 5-fold CV over all 300 silver sentences |
| `results/phase3/ner_test_predictions.tsv` | Token-level silver tags vs. every system on silver_test, for error analysis |
| `results/phase3/ner_token_confusion_matrix.csv` | Token-level BIO confusion matrix for the CRF baseline; diagnostic, not entity-level scoring |

Reproduce (Python 3.12 venv with `scripts/phase3/requirements-ner.txt`):

```bash
PY=.venv-phase2/bin/python
$PY scripts/phase3/sample_ner_sentences.py \
    --sentences "$PWD/data/processed/combined/sentences.tsv" \
    --documents "$PWD/data/processed/combined/documents.jsonl" \
    --n 3000 --out data/eval/ner/pool.conll
$PY scripts/phase3/build_ner_silver.py
$PY scripts/phase3/ner_baselines.py
```

Rerunning the sampler after the corpus is regenerated gives a **different pool**. The sentence IDs in `silver_spans.txt` (`ner00001`...) refer to the committed `pool.conll`. Do not regenerate the pool once annotation has started. Each pool sentence records its source as `doc_id:sent_id` and keeps its original text in `# text =`.

## 2. Sampling

Sentences with 6–40 tokens and ≥ 90% Devanagari letters are kept as candidates. Each category's share is proportional to √(number of candidates), with a floor of 150 sentences. News is still the largest share, but literature, culture and education stay visible. The sampled sentences are interleaved, so every prefix of the pool has the same category mix. That is why the first 300 sentences make a fair silver subset.

| Category | Candidates | In pool (3,000) | In silver (300) |
|---|---:|---:|---:|
| 01_literature | 17,760 | 339 | 34 |
| 02_news_media | 312,440 | 1,500 | 150 |
| 03_education_science | 307 | 150 | 15 |
| 04_culture_history | 10,694 | 257 | 26 |
| 06_web_articles | 81,699 | 754 | 75 |

Categories 05 and 07 have no documents in the current processed corpus.

## 3. Silver annotation (AI first pass)

| Split | Sentences | Tokens | Sentences with ≥ 1 entity | PER | LOC | ORG | MISC | DATE | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 300 | 3,595 | 143 | 114 | 114 | 37 | 54 | 31 | 350 |
| dev | 151 | 1,824 | 77 | 52 | 55 | 13 | 33 | 14 | 167 |
| test | 149 | 1,771 | 66 | 62 | 59 | 24 | 21 | 17 | 183 |

Entity density differs a lot by domain. Literature averages 0.32 entities per sentence, and only 6 of its 34 sentences have any entity. News averages 1.31 and web articles 1.47 (mostly Wikipedia, with many places and dates). Education is at 0.73 and culture/history at 0.85. A gold set drawn only from literature would be almost empty, so the 2–3k gold set should keep the news and web share.

## 4. Baselines (on silver_test, 149 sentences / 183 entities)

Entity-level scores from seqeval, conlleval-style exact span and type match.

| System | Training data | P | R | F1 |
|---|---|---:|---:|---:|
| `gaz_seed`: seed gazetteer + DATE/honorific rules | none | 0.786 | 0.361 | **0.494** |
| `gaz_seed_dev`: + every entity string in silver_dev | dev lexicon | 0.791 | 0.372 | **0.506** |
| `crf_dev`: CRF, c1=0.05 c2=0.01 (3-fold CV on dev) | 151 silver sentences | 0.742 | 0.268 | 0.394 |
| `crf_dev_weak`: CRF on dev + 1,500 pool sentences labelled by `gaz_seed_dev` | weak supervision | 0.786 | 0.361 | 0.494 |

Per type (F1). Full P/R/support is in `results/phase3/ner_scores.csv`.

| System | PER | LOC | ORG | MISC | DATE |
|---|---:|---:|---:|---:|---:|
| gaz_seed | 0.16 | 0.65 | 0.50 | 0.52 | 0.78 |
| gaz_seed_dev | 0.16 | 0.64 | 0.50 | 0.61 | 0.82 |
| crf_dev | 0.06 | 0.65 | 0.14 | 0.61 | 0.32 |
| crf_dev_weak | 0.14 | 0.64 | 0.50 | 0.61 | 0.72 |

5-fold cross-validation over all 300 silver sentences (`ner_cv.csv`): micro F1 is 0.500 for the seed gazetteer, 0.496 for the gazetteer plus training-fold lexicon, and 0.482 for the CRF. Across folds the F1 ranges from 0.36 to 0.62, so with this little data the differences between systems are within noise.

### Observations
- **PER is the main gap.** Recall is under 10% for every system. Konkani person names are open-class and do not repeat across 150 sentences, so a lexicon cannot cover them. The honorific (`डॉ`, `प्रा` ...) and `हांणी/हांचे` trigger rules catch only a few names, and the trigger rule also fires on non-names (`धर्म वेवसाय हांचे`). Names in lists (cricket line-ups, cast lists) and epic characters are missed entirely.
- **LOC and DATE are easy.** A short Goa/India place list with suffix stripping (`गोंयांत` → `गोंय`) gives LOC F1 of 0.65. Day + month + year patterns give DATE F1 of about 0.8.
- **The CRF is limited by training size.** With only 151 sentences it learns little beyond the gazetteer features and loses recall. Adding 1,500 weakly labelled sentences brings it back to the gazetteer's level, but not above it, because it learns to copy the gazetteer. A larger gold training set (FR-P3-2: 2–3k sentences) is needed before the CRF, BiLSTM-CRF or MuRIL can be compared fairly.
- **Precision is fairly high (~0.78) and recall low (~0.36).** The gazetteer is a useful pre-annotation tool: it marks some entities correctly and misses many, so annotators mostly add spans rather than delete them.
- **The baseline numbers are optimistic.** The seed gazetteer was written by the same annotator who labelled the silver set. Entries clearly prompted by test sentences were removed, but some bias remains. The silver labels and the gazetteer also share one person's judgement.

## 5. What the team must do
1. **Verify the 300 silver sentences.** Check every line of `data/eval/ner/silver_spans.txt` (or import `silver_all.conll` into Doccano/Label Studio). Pay special attention to the weak spots listed in `docs/ner_guidelines.md` §6: deities as PER, dynasties as ORG, MISC titles, garbled OCR sentences, and names left O. Then rerun `build_ner_silver.py` and `ner_baselines.py`.
2. Confirm or change the open guideline decisions: deities → PER; demonyms and languages → MISC; country-as-team → LOC; dynasties → ORG; whole token tagged with its case suffix.
3. Annotate the rest of the pool toward 2–3k sentences, with ≥ 200 shared sentences for Cohen's κ (target ≥ 0.7).
4. Grow `gazetteer_seed.tsv` (Goa villages, common Goan surnames, institutions). This raises pre-annotation recall.
5. After gold data exists, add the BiLSTM-CRF (with our FastText vectors) and the fine-tuned MuRIL/IndicBERT model, both evaluated on the same test split.

## 6. Limitations
- The silver labels have no inter-annotator agreement and may contain systematic errors from a single annotator.
- The test set has 183 entities, so one entity is worth about 0.5 F1 points. ORG (24) and DATE (17) per-type scores are especially unstable.
- Punctuation is removed by the corpus tokeniser. Abbreviations such as `एल आय सी` therefore appear as separate tokens, and the sentence context loses its punctuation cues.

## 7. Experimental BiLSTM-CRF implementation

`scripts/phase3/ner_bilstm_crf.py` now provides a word/suffix BiLSTM with a
linear-chain CRF, BIO transition constraints, optional local FastText vectors,
and `train` / `predict` commands. The training command uses an internal
validation slice from `silver_dev.conll`, keeps `silver_test.conll` out of model
selection, saves a versioned checkpoint atomically, and writes entity-level
P/R/F1 plus a token-level BIO confusion matrix. The existing CRF baseline also
adds two-token context features and writes a token-level confusion matrix for
its CRF predictions.

Example (Windows, from the repository root):

```powershell
.venv-phase2\Scripts\python.exe scripts\phase3\ner_bilstm_crf.py train
.venv-phase2\Scripts\python.exe scripts\phase3\ner_bilstm_crf.py predict `
    --checkpoint data\models\phase3_ner_bilstm_crf.pt `
    --input data\eval\ner\silver_test.conll `
    --out results\phase3\ner_bilstm_crf_predictions.conll
```

An optional local FastText KeyedVectors file can be supplied with
`--fasttext data\models\ft25.kv`; its dimension must match `--embedding-dim`.
This is an experimental implementation only. No NER model was trained as part
of the development update, and no new scores are claimed. Existing and future
scores on `silver_*` files describe agreement with AI-generated, unverified
labels; they are not human-gold results. Annotate and review a genuine gold
test set before using these figures as evidence of NER quality.
