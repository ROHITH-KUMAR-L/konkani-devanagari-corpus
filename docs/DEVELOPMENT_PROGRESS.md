# Development progress

Updated: 2026-11-22

## Completed in this update

- Read the requested project overview, Phase 1 report, Phase 3 neural LM and NER documentation, and NER annotation guidelines.
- Inspected the Phase 3 LM, NER baselines/common code, corpus utilities, demo endpoints, and demo UI.
- Added an experimental word/suffix BiLSTM-CRF with BIO-constrained transitions, optional local FastText initialization, internal validation, separate test evaluation, inference, and atomic versioned checkpoints.
- Added explicit annotation-status labeling to the experimental model outputs. Existing silver labels are still identified as AI-generated and not human-verified gold.
- Added token-level BIO confusion-matrix output for the existing CRF baseline and the experimental BiLSTM-CRF; entity-level precision, recall, and F1 remain the primary NER metrics.
- Added CoNLL row/tag validation and additional morphology/script/context features for the CRF.
- Added atomic, versioned neural-LM checkpoints with legacy state-dict loading, model/vocabulary compatibility checks, training-split guards, and positive training-configuration validation.
- Improved demo input validation, missing-model error ordering, and HTTP status codes for malformed input, missing files, model vocabulary errors, and internal failures.
- Added focused unit tests for corpus tokenisation, segmentation, CoNLL handling, gazetteer suffix matching, model loss/decode/checkpoint round trips, and demo input/missing-file handling.
- Updated [README.md](../README.md) and [phase3_ner.md](phase3_ner.md) to describe the experimental implementation and silver-data limitation.
- Updated [phase3_neural_lm.md](phase3_neural_lm.md) with checkpoint format and training guard behavior.

## Files changed

- `scripts/phase3/neural_lm.py`
- `scripts/phase3/ner_common.py`
- `scripts/phase3/ner_baselines.py`
- `scripts/phase3/ner_bilstm_crf.py` (new)
- `scripts/demo/app.py`
- `scripts/demo/index.html`
- `tests/test_project_utils.py` (new)
- `tests/test_model_utilities.py` (new)
- `tests/test_demo.py` (new)
- `docs/phase3_ner.md`
- `docs/phase3_neural_lm.md`
- `README.md`
- `docs/DEVELOPMENT_PROGRESS.md` (new)

## Environment and data checks

- The user installed the Phase 2/3 Python dependencies in `.venv-phase2`; the environment now runs the focused tests.
- Dependency versions were not audited independently.
- Repository file inspection did not find `data/processed/combined/`, `data/splits/{train,dev,test}.txt`, `data/models/`, the SentencePiece model, or a Phase 3 LSTM checkpoint. The team documentation says the processed corpus and model files are supplied separately and are git-ignored.
- The AI-annotated NER `silver_all.conll`, `silver_dev.conll`, and `silver_test.conll` files and seed gazetteer are present. They are not human-verified gold and should not be used to claim gold performance.
- No corpus, pretrained vector, SentencePiece model, or language-model checkpoint was fabricated.

## Phase 3 review notes

- The existing neural-LM write-up reports that the time-boxed neural runs had not reached early stopping and that the longer LSTM/Transformer runs were still improving. These are existing recorded measurements, not measurements from this update. Re-running or comparing models is blocked by the absent splits and model assets.
- The previous LM model designs and evaluation protocol were left intact. This update changes checkpoint safety and empty-data/configuration handling rather than retuning architectures.
- NER currently has gazetteer/rule and CRF baselines on only 300 AI-first-pass sentences, with no human agreement measure. The new BiLSTM-CRF is experimental and has not been trained.
- The demo's language and sentence utilities are independent of the model archive, but tokenisation, autocomplete, similarity, generation, and corpus search require ignored corpus/model files that are absent here.

## Tests and measurements

- VS Code diagnostics report no errors in the modified Python files. This is not a substitute for execution.
- `.venv-phase2\Scripts\python.exe -m unittest discover -s tests -v`: **14 tests passed**.
- `.venv-phase2\Scripts\python.exe -m compileall -q scripts\phase3 scripts\demo`: passed.
- Dependency version checks and full demo integration remain unverified.
- No NER model was trained and no new evaluation results were measured. Existing Phase 3 results in the repository are unchanged.
- Demo endpoints that depend on the absent processed corpus, tokenizers, embedding vectors, or language-model checkpoint could not be integration-tested.

## Bugs and reliability changes

- Demo search now rejects empty queries instead of scanning the corpus for an empty string.
- Demo similarity/analogy rejects requests without a word or a complete analogy.
- Missing LSTM checkpoint and SentencePiece files are reported before attempting unrelated model imports.
- Demo API errors no longer report missing model files as HTTP 200; malformed input returns 400, missing files 503, unknown vocabulary terms 422, and unexpected failures 500 with server-side logging.
- LM checkpoint writes are atomic; versioned checkpoints verify model type and vocabulary size while old plain state-dict checkpoints remain readable.
- LM training now fails explicitly on empty training/development splits and validates key positive settings.

## Remaining blockers and next steps

1. Obtain the team-provided processed-corpus and model archives; confirm split counts, SentencePiece vocabulary, and model configuration before LM training.
2. Have humans review the 300 silver sentences and annotate a genuine gold set with inter-annotator agreement. Only then evaluate and report NER quality as gold performance.
3. After data and configuration verification, run experimental NER training/inference and compare it with the rule and CRF baselines. Do not treat silver-only scores as evidence of generalization.
4. Smoke-test all demo tabs once their required files are available.
