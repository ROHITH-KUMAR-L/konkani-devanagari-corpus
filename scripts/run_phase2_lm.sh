#!/usr/bin/env bash
# Phase 2 (part A): tokenisers + n-gram language models, end to end.
# Needs data/splits/{train,dev,test}.txt (from scripts/run_phase1.sh) and the
# Phase 2 environment (.venv-phase2 with sentencepiece, numpy, matplotlib).
# Override the interpreter with PYTHON=/path/to/python.
#   bash scripts/run_phase2_lm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv-phase2/bin/python}"

"$PY" scripts/phase2/tokenizers.py --splits data/splits --models data/models --out results/phase2
"$PY" scripts/phase2/ngram_lm.py --selftest --splits data/splits
"$PY" scripts/phase2/ngram_lm.py --splits data/splits --out results/phase2 \
    --spm data/models/spm_unigram_16000.model
