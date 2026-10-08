#!/usr/bin/env bash
# Reproduce everything from this folder (Apple-Silicon GPU via MPS, CUDA or CPU).
#   bash run.sh --smoke   # ~1 min: 3 runs x 2 epochs into results_smoke/
#   bash run.sh           # all 51 training runs (~3.5 h on an M2 Pro), resumable, then report + figures
# Set PY=/path/to/python to use an existing environment; otherwise a local .venv is created.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
REPO="${REPO:-..}"

if [ -z "${PY:-}" ]; then
  if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
  fi
  PY=.venv/bin/python
fi
$PY -c "import torch; print('torch', torch.__version__, '| MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"

$PY build_manifest.py --repo "$REPO" > manifest_audit.txt
$PY run_experiments.py --repo "$REPO" --smoke
if [ "${1:-}" = "--smoke" ]; then echo "Smoke test passed."; exit 0; fi

$PY msssim_baseline.py --repo "$REPO" --out results
CAFF=""; command -v caffeinate >/dev/null && CAFF="caffeinate -i"
$CAFF $PY run_experiments.py --repo "$REPO" 2>&1 | tee -a results/training_log.txt
$PY analyze.py --results results --out report
$PY make_figures.py --results results --out figures
$PY make_fig5.py
$PY make_fig15.py --repo "$REPO" --errors --out figures/fig15_qualitative_with_errors
$PY make_fig15.py --repo "$REPO" --out figures/fig15_qualitative
echo "Done: see report/REPORT.md and figures/."
