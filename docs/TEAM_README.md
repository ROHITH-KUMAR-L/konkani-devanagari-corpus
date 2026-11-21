# Konkani corpus: processed data for the team

This zip holds the cleaned corpus and the train/dev/test splits. They are not on GitHub because they are large (about 870 MB unzipped) and contain copyrighted newspaper and book text. **Do not upload or share it outside the team.**

Snapshot: 21 Nov 2026, **6.06M clean tokens**, 329 documents, 5 domains (news 75.7%, web/Wikipedia 17.8%, literature 3.9%, culture 2.6%, education 0.1%). All text is UTF-8, Unicode NFC, LF line endings.

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

## 4. What the team still has to check

These were first-pass labelled by Claude (an AI). Fix them before quoting their numbers:

- `data/eval/segmentation_gold.txt`, `segmentation_gold_heldout.txt`: sentence boundaries
- `data/eval/language_audit_sample.tsv`: Konkani / not Konkani
- `data/eval/word_similarity.tsv`, `analogies.tsv`: embedding test sets
- `data/eval/ner/silver_spans.txt`: 300 NER sentences (rules in `docs/ner_guidelines.md`)

## 5. Newer data

The e-paper collection later grew to 573 issues; this zip has the first 289. To include all of them, rebuild instead of using the zip: `bash scripts/run_phase1.sh` (or `scripts/rebuild_all.sh` for Phase 1 + Phase 2 + a new zip). A newer zip will be shared after that run.
