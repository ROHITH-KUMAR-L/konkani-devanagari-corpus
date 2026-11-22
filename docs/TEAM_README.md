# Konkani corpus: data and models for the team

The data is shared as two zips in the team's Google Drive folder ("Konkani NLP project data"). It is not on GitHub: seven files are over GitHub's 100 MB limit, and the newspaper and book text is copyrighted. **Do not upload or share the zips outside the team.**

| Zip | Size | Contains | Who needs it |
|---|---|---|---|
| `konkani_corpus_processed.zip` | 268 MB (about 1.5 GB unzipped) | Clean corpus, sentences, train/dev/test splits, report, data dictionary, this guide | Everyone |
| `konkani_models.zip` | 717 MB | Trained models: SentencePiece tokenisers, word vectors (word2vec, FastText, GloVe), neural LMs (FFNN, LSTM, Transformer) | Anyone who runs the demo or uses the models without retraining (about an hour) |

Snapshot: 22 Nov 2026 (`corpus-v1.0`), **10.51M clean tokens**, 613 documents, 5 domains (news 86.0%, web/Wikipedia 10.3%, literature 2.2%, culture 1.5%, education 0.05%). Splits: train 8.46M, dev 1.01M, test 1.04M tokens. All text is UTF-8, Unicode NFC, LF line endings.

## 1. Set up

```bash
git clone https://github.com/ROHITH-KUMAR-L/konkani-devanagari-corpus.git
cd konkani-devanagari-corpus
unzip ~/Downloads/konkani_corpus_processed.zip      # answer "A" (all) if asked to overwrite
unzip ~/Downloads/konkani_models.zip                # optional
```

Unzip **inside the repo folder**. Afterwards these folders must exist: `data/processed/`, `data/splits/` and (with the models zip) `data/models/`.

* **Windows (Extract All):** set the destination to the `konkani-devanagari-corpus` folder itself. Do not let Windows create a new folder named after the zip.
* The corpus zip also contains `results/split_summary.json`, `metadata/data_dictionary.md` and `docs/report.md`, which are the same as in git; overwriting them is fine.
* Code changes: just `git pull`. The zips only need to be downloaded again when a new data version is announced.
* Delete any older copy of the corpus zip (it had empty folders for categories 05 and 07, which have no data).

## 2. Which file to use

| You want to | Use |
|---|---|
| Train or evaluate any model (LMs, embeddings, tokenisers) | `data/splits/train.txt`, `dev.txt`, `test.txt`: one sentence per line, split by document (80/10/10, seed 2026), so no document is in two splits |
| Train on a less news-heavy mix | `data/splits/balanced/{train,dev,test}.txt` (no domain above 60% of tokens) |
| Know which document each split sentence came from | `data/splits/split_manifest.csv` |
| Read the whole clean corpus | `data/processed/combined/konkani_devanagari_clean.txt` (one paragraph per line) |
| Sentences with document id | `data/processed/combined/sentences.tsv` (`doc_id`, `sent_id`, `n_tokens`, `sentence`) |
| Whole documents | `data/processed/combined/documents.jsonl` (`{"doc_id", "category", "text"}`) |
| One domain only | `data/processed/by_category/<category>/corpus_clean.txt` or `sentences.tsv` (01, 02, 03, 04, 06) |

Every column and file is described in `metadata/data_dictionary.md`. The numbers and cleaning steps are in `docs/report.md`.

Always report test results on `test.txt`. Tune on `dev.txt` only.

## 3. Python environments

Two environments are used. `uv` (`pip install uv`) is fastest; plain `python -m venv` + `pip install` works too.

```bash
# Phase 1 (any recent Python)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Phase 2/3 and the demo: Python 3.12 (gensim does not build on 3.14)
uv venv -p 3.12 .venv-phase2
uv pip install -p .venv-phase2 gensim sentencepiece scikit-learn numpy scipy matplotlib sklearn-crfsuite seqeval
uv pip install -p .venv-phase2 torch --index-url https://download.pytorch.org/whl/cpu
```

On Windows replace `.venv-phase2/bin/python` with `.venv-phase2\Scripts\python` in the commands below.

## 4. Run the live demo

Needs the models zip and `.venv-phase2`:

```bash
.venv-phase2/bin/python scripts/demo/app.py --warm      # then open http://127.0.0.1:8050
```

Tabs: Overview, Language check, Sentences, Corpus search, Tokenisers, Autocomplete, Similar words, Generate (LSTM), Names (NER), plus an on-screen Konkani keyboard. Use `--port 8060` if port 8050 is busy.

**English translation (optional).** The "Translate to English" buttons use AI4Bharat's IndicTrans2, an outside pretrained model:

1. `uv pip install -p .venv-phase2 "transformers<4.50" IndicTransToolkit`
2. Create a Hugging Face account and click **Agree** on https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M (the model is gated).
3. Create a Read token at https://huggingface.co/settings/tokens and run `.venv-phase2/bin/hf auth login`.

The first translation downloads about 1 GB. Every other tab works without this.

## 5. Re-train instead of using the models zip

```bash
bash scripts/run_phase2_lm.sh            # tokenisers + n-gram LMs      (~3 min)
bash scripts/run_phase2_embeddings.sh    # word2vec / FastText / GloVe   (~25 min)
bash scripts/run_phase3_lm.sh            # neural LMs, installs torch CPU (~25 min)
```

Results go to `results/phase2/` and `results/phase3/`. Write-ups: `docs/phase2_lm.md`, `docs/phase2_embeddings.md`, `docs/phase3_neural_lm.md`, `docs/phase3_ner.md`.

## 6. What the team still has to check

These were labelled by Claude (an AI) and reviewed by a second AI pass, not yet by a person. Spot-check them before quoting their numbers:

- `data/eval/segmentation_gold.txt`, `segmentation_gold_heldout.txt`: sentence boundaries (review notes in `data/eval/review/`)
- `data/eval/language_audit_sample.tsv`: Konkani / not Konkani
- `data/eval/word_similarity.tsv`, `analogies.tsv`: embedding test sets
- `data/eval/ner/silver_spans.txt`: 300 NER sentences (rules in `docs/ner_guidelines.md`)
- `data/eval/ocr_gold/epaper/`: newspaper OCR reference text

## 7. Version

Final Phase 1 corpus: all 573 Bhaangarbhuin issues (2 Apr 2025 – 20 Nov 2026; later issues are behind a paywall). It matches git tag `corpus-v1.0`. To rebuild it from the committed OCR text: `bash scripts/run_phase1.sh`. Phase 2 and 3 results in `docs/` and `results/` were produced on these same splits.
