#!/usr/bin/env bash
# Phase 2 (part B): train and evaluate Konkani word embeddings end to end.
#   1. word2vec CBOW / skip-gram, FastText (3-6 and 2-5 char n-grams), GloVe on data/splits/train.txt
#   2. learning-curve models (skip-gram, FastText) on 10/25/50% of the training sentences
#   3. intrinsic evaluation, nearest neighbours, t-SNE plots, OOV check -> results/phase2/
# Needs the Phase 2 environment (gensim, numpy, scipy, scikit-learn, matplotlib):
#   PYTHON=.venv-phase2/bin/python bash scripts/run_phase2_embeddings.sh
# Models go to data/models/ (git-ignored). Training uses 4 worker threads.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv-phase2/bin/python ]; then PYTHON=.venv-phase2/bin/python; else PYTHON=python; fi
fi

start=$(date +%s)

"$PYTHON" scripts/phase2/train_embeddings.py --train data/splits/train.txt --out data/models
"$PYTHON" scripts/phase2/train_embeddings.py --train data/splits/train.txt --out data/models \
    --models sg ft --fractions 0.1 0.25 0.5
"$PYTHON" scripts/phase2/eval_embeddings.py --models data/models \
    --sim data/eval/word_similarity.tsv --analogies data/eval/analogies.tsv --out results/phase2

echo "run_phase2_embeddings.sh finished in $(( $(date +%s) - start ))s"
