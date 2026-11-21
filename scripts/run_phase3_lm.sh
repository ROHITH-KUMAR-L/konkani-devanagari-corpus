#!/usr/bin/env bash
# Phase 3 (FR-P3-1): neural language models, compared with the Phase 2 n-gram LMs.
# Needs data/splits/{train,dev,test}.txt (from scripts/run_phase1.sh) and the
# Phase 2 environment (.venv-phase2 with sentencepiece, numpy, matplotlib).
# PyTorch (CPU wheel) is installed into that environment if it is missing.
#   bash scripts/run_phase3_lm.sh                 # about 30 min on a laptop CPU
#   SPLITS=/abs/path/to/splits bash scripts/run_phase3_lm.sh
#   bash scripts/run_phase3_lm.sh --models lstm --budget lstm=20   # extra args go to neural_lm.py
# Override the interpreter with PYTHON=/path/to/python.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv-phase2/bin/python}"
SPLITS="${SPLITS:-data/splits}"
SPM=data/models/spm_unigram_16000.model

if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "installing PyTorch (CPU wheel) into $("$PY" -c 'import sys; print(sys.prefix)')"
    if "$PY" -m pip --version >/dev/null 2>&1; then
        "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    else    # uv-created venvs have no pip
        uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
    fi
fi
"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# 1. SentencePiece Unigram-16k with the Phase 2 settings (skipped if it exists)
"$PY" scripts/phase3/neural_lm.py --prepare-only --splits "$SPLITS" --spm "$SPM"
# 2. n-gram baselines on the same splits (word + subword modified KN; ~3 min)
"$PY" scripts/phase2/ngram_lm.py --splits "$SPLITS" --out results/phase3/ngram_baseline \
    --spm "$SPM" --skip curve
# 3. neural LMs (default budgets: 2.5 + 7 + 7 + 3.5 min training, plus test scoring)
"$PY" scripts/phase3/neural_lm.py --splits "$SPLITS" --spm "$SPM" --out results/phase3 \
    --baseline-dir results/phase3/ngram_baseline "$@"
