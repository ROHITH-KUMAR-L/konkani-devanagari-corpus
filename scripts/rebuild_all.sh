#!/usr/bin/env bash
# Rebuild everything downstream of the collected data, then pack the teammate zip.
#   Phase 1 pipeline (.venv)  ->  Phase 2 LM + embeddings (.venv-phase2)  ->  ../konkani_corpus_processed.zip
# Usage: bash scripts/rebuild_all.sh            (writes rebuild_all.log, and rebuild_all.done on success)
#        WAIT_FOR_FETCH=1 bash scripts/rebuild_all.sh   (first wait until fetch_epaper.py fetch exits)
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f rebuild_all.done

if [ "${WAIT_FOR_FETCH:-0}" = 1 ]; then
    while pgrep -f "fetch_epaper.py fetch" >/dev/null; do sleep 300; done
    echo "fetch finished $(date)"
fi

echo "== phase 1 $(date)"
PYTHON=.venv/bin/python PYTHONWARNINGS=ignore bash scripts/run_phase1.sh
echo "== phase 2 LM $(date)"
bash scripts/run_phase2_lm.sh
echo "== phase 2 embeddings $(date)"
bash scripts/run_phase2_embeddings.sh

echo "== zip $(date)"
ZIP=../konkani_corpus_processed.zip
rm -f "$ZIP"
zip -qr9 "$ZIP" data/processed data/splits results/split_summary.json metadata/data_dictionary.md docs/report.md
unzip -tq "$ZIP"
ls -lh "$ZIP"

date > rebuild_all.done
echo "== done $(date)"
