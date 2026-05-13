#!/bin/bash
# ============================================================
# run_pipeline.sh
# Master Execution Script for IML Forecasting Pipeline
# ============================================================

set -e # Exit immediately if a command exits with a non-zero status.

echo "=========================================================="
echo "Starting Full Pipeline Execution"
echo "=========================================================="

# Activate Virtual Environment
source ../.venv/bin/activate

# 1. Train Base Model
echo -e "\n[1/7] Training Base CatBoost Model..."
python training/train_catboost.py

# 2. Train HMM Corrector
echo -e "\n[2/7] Training HMM Residual Corrector..."
python training/train_hmm_correction.py

# 3. Build Residual Sequences for Residual Models
echo -e "\n[3/7] Building Sequence Tensors..."
python preprocessing/build_residual_sequences.py

# 4. Train Residual Models
echo -e "\n[4/7] Training Residual Models (Transformer/MLP/CNN/Linear/CatBoost)..."
python training/train_residual_models.py

# 5. Evaluate on Test Split
echo -e "\n[5/7] Evaluating Models on Test Split..."
python evaluation/evaluate_test.py

# 6. Evaluate on Holdout Set
echo -e "\n[6/7] Evaluating Models on Holdout Companies..."
python evaluation/evaluate_holdout.py

# 7. Generate Dashboards
echo -e "\n[7/7] Generating Analytics Dashboards..."
python evaluation/generate_dashboards.py --mode all

echo -e "\n=========================================================="
echo "PIPELINE COMPLETE!"
echo "Check results/dashboards/ for visual outputs."
echo "=========================================================="
