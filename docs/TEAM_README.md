# Konkani corpus: processed data for the team

This zip holds the cleaned corpus and the train/dev/test splits. They are not on GitHub because they are large (about 1.5 GB unzipped) and contain copyrighted newspaper and book text. **Do not upload or share it outside the team.**

Snapshot: 22 Nov 2026 (`corpus-v1.0`), **10.51M clean tokens**, 613 documents, 5 domains (news 86.0%, web/Wikipedia 10.3%, literature 2.2%, culture 1.5%, education 0.05%). Splits: train 8.46M, dev 1.01M, test 1.04M tokens. Because news dominates, consider the balanced split (below) for anything domain-sensitive. All text is UTF-8, Unicode NFC, LF line endings.

## 1. Set up

```bash
git clone https://github.com/ROHITH-KUMAR-L/konkani-devanagari-corpus.git
cd konkani-devanagari-corpus
unzip /path/to/konkani_corpus_processed.zip        # run from the repo root
```

The zip unpacks into `data/processed/` and `data/splits/`, the folders the scripts expect. It also contains `results/split_summary.json`, `metadata/data_dictionary.md` and `docs/report.md`, which are the same as in git. Let unzip overwrite them, or skip them.

## 2. Which file to use

| You want to | Use |
|---|---|
| Train or evaluate any model (LMs, embeddings, tokenisers) | `data/splits/train.txt`, `dev.txt`, `test.txt`: one sentence per line, split by document (80/10/10, seed 2026), so no document is in two splits |
| Train on a less news-heavy mix | `data/splits/balanced/{train,dev,test}.txt` (no domain above 60% of tokens) |
| Know which document each split sentence came from | `data/splits/split_manifest.csv` |
| Read the whole clean corpus | `data/processed/combined/konkani_devanagari_clean.txt` (one paragraph per line) |
| Sentences with document id | `data/processed/combined/sentences.tsv` (`doc_id`, `sent_id`, `n_tokens`, `sentence`) |
| Whole documents | `data/processed/combined/documents.jsonl` (`{"doc_id", "category", "text"}`) |
| One domain only | `data/processed/by_category/<category>/corpus_clean.txt` or `sentences.tsv` |

Every column and file is described in `metadata/data_dictionary.md`. The numbers and cleaning steps are in `docs/report.md`.

Always report test results on `test.txt`. Tune on `dev.txt` only.

## 3. Run the project code

Two Python environments are used:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Phase 1 (any recent Python)
# Phase 2/3 need Python 3.12 (gensim does not build on 3.14):
uv venv -p 3.12 .venv-phase2 && uv pip install -p .venv-phase2 gensim sentencepiece scikit-learn numpy matplotlib
```

Then:

```bash
bash scripts/run_phase2_lm.sh            # tokenisers + n-gram LMs   (~2 min)
bash scripts/run_phase2_embeddings.sh    # word2vec / FastText / GloVe (~25 min)
bash scripts/run_phase3_lm.sh            # neural LMs, installs torch CPU (~25 min)
```

Results go to `results/phase2/` and `results/phase3/`. The write-ups are `docs/phase2_lm.md`, `docs/phase3_neural_lm.md` and `docs/phase3_ner.md`.

## 3b. Trained models and the live demo (optional)

`konkani_models.zip` (717 MB, shared separately) holds the trained models: SentencePiece tokenisers, word vectors, neural LMs. Unzip it from the repo root as well; it fills `data/models/`. Then:

```bash
.venv-phase2/bin/python scripts/demo/app.py --warm      # open http://127.0.0.1:8050
```

Without the models zip, run the three scripts above first (about an hour). English translation in the demo also needs `uv pip install -p .venv-phase2 "transformers<4.50" IndicTransToolkit` and a Hugging Face login (accept the terms of `ai4bharat/indictrans2-indic-en-dist-200M`, then `.venv-phase2/bin/hf auth login`).

## 4. What the team still has to check

These were first-pass labelled by Claude (an AI). Fix them before quoting their numbers:

- `data/eval/segmentation_gold.txt`, `segmentation_gold_heldout.txt`: sentence boundaries
- `data/eval/language_audit_sample.tsv`: Konkani / not Konkani
- `data/eval/word_similarity.tsv`, `analogies.tsv`: embedding test sets
- `data/eval/ner/silver_spans.txt`: 300 NER sentences (rules in `docs/ner_guidelines.md`)

## 5. Version

This is the final Phase 1 corpus: all 573 Bhaangarbhuin issues (2 Apr 2025 – 20 Nov 2026; later issues are behind a paywall). It matches git tag `corpus-v1.0`. To rebuild it from the committed OCR text: `bash scripts/run_phase1.sh`.

Note: the Phase 2/3 numbers in `docs/` were measured on the earlier 6.06M-token splits; they will be refreshed after a re-run on these splits.
