#!/usr/bin/env bash
# Run the whole Phase 1 pipeline from data/raw + OCR caches to results and splits.
# OCR / downloads are separate, slow, cached steps (see README "How to run"):
#   python scripts/fetch_epaper.py fetch
#   python scripts/fetch_epaper.py ocr --pdf-dir data/raw/03_education_science \
#       --out data/interim/03_education_science --dpi 300 --force-ocr
#   python scripts/extract_ldcil.py ...
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

$PY scripts/extract_text.py --out data/interim/iter0
$PY scripts/clean_text.py      --in data/interim/iter0 --out data/interim/iter1
$PY scripts/validate_konkani.py --in data/interim/iter1 --out data/interim/iter2
$PY scripts/deduplicate.py     --in data/interim/iter2 --out data/interim/iter3

$PY scripts/analyze_corpus.py --in data/interim/iter0 --out results/iter0 --label "iteration 0 (raw)"
$PY scripts/analyze_corpus.py --in data/interim/iter1 --out results/iter1 --label "iteration 1 (cleaned)"
$PY scripts/analyze_corpus.py --in data/interim/iter2 --out results/iter2 --label "iteration 2 (Konkani only)"
$PY scripts/analyze_corpus.py --in data/interim/iter3 --out results --label "final (deduplicated)"
$PY scripts/document_audit.py --iters data/interim/iter0 data/interim/iter1 data/interim/iter2 data/interim/iter3

$PY scripts/segment_sentences.py --in data/interim/iter3 --out data/processed
$PY scripts/segment_sentences.py --eval data/eval/segmentation_gold.txt --eval-out results/segmentation_eval.txt
$PY scripts/segment_sentences.py --eval data/eval/segmentation_gold_heldout.txt --eval-out results/segmentation_eval_heldout.txt
$PY scripts/length_analysis.py --processed data/processed --out results
$PY scripts/make_splits.py --processed data/processed --out data/splits
