#!/usr/bin/env bash
# Run diagnostics plotting without retraining models
set -euo pipefail

PY=python3
SCRIPT=scripts/diagnostics/plot_company_overlay.py
COMPANY=${1:-5048}

echo "Running diagnostics for company ${COMPANY}"
${PY} ${SCRIPT} --company ${COMPANY} \
    --company-csv holdout_set/${COMPANY}.csv \
    --predictions results/rmse/holdout/predictions.csv results/rmse/test/predictions.csv \
    --out results/diagnostics/company_${COMPANY}_overlay.png

echo "Done. Output: results/diagnostics/company_${COMPANY}_overlay.png"
