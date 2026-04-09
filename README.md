IML project

CatBoost + residual CNN pipeline for monthly return prediction using a
company-month panel dataset.

## Repository layout

- documentation/ : LaTeX and PDFs for the writeups (preprocessing and ML).
- preprocess_factors.py : preprocessing script for factor_data.csv.
- impliment.ipynb : end-to-end training + inference notebook.
- processed_data/ : generated splits and metadata (train/validation/test).
- testing results company wise/ : per-company plots and test metrics CSVs.
- 194.csv : holdout company file used for standalone inference.

Other folders (optional, not required to run the pipeline):

- catboost_info/ : CatBoost training logs.
- bemer/ : presentation sources and build artifacts.
- old data/ : legacy plots and prior result exports.

## Processed data outputs

After preprocessing, outputs are written to processed_data/:

- processed_full.csv
- train.csv
- validation.csv
- test.csv (targets masked for leakage-safe inference)
- test_targets.csv (true targets for evaluation)
- metadata.json

## Documentation

See documentation/ for:

- preprocess.tex / preprocess.pdf
- ML.tex / ML.pdf
- dataset_description.pdf

## Per-company results

Per-company plots and evaluation outputs are written to:

- testing results company wise/ (company_*.png, test_predictions.csv,
	company_metrics.csv, test_metrics_summary.csv)

## How to run

### 1) Preprocess the raw data

Time-based split (default):

```
python preprocess_factors.py \
	--input factor_data.csv \
	--output-dir processed_data \
	--split-mode time \
	--train-end-month 2019-09 \
	--validation-end-month 2023-09 \
	--split-date-source month
```

Company-based split:

```
python preprocess_factors.py \
	--input factor_data.csv \
	--output-dir processed_data \
	--split-mode company \
	--train-share 0.8 \
	--validation-share 0.1 \
	--test-share 0.1 \
	--validation-min-history 6 \
	--random-seed 13
```

### 2) Run the notebook

Open impliment.ipynb in VS Code or Jupyter and run cells top-to-bottom.
The notebook reads processed_data/ and writes results to
testing results company wise/.

If packages are missing, install (CPU example):

```
pip install pandas numpy catboost torch ipython plotly
```
