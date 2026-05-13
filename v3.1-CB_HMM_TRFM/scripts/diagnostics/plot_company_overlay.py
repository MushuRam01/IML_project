#!/usr/bin/env python3
"""
Plot a company's actual returns and overlay model predictions (test + holdout).

Usage:
  python scripts/diagnostics/plot_company_overlay.py --company 5048

Writes: results/diagnostics/company_<id>_overlay.png
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_company_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Month"]) if path.exists() else None
    return df


def load_predictions(pred_paths):
    dfs = []
    for p in pred_paths:
        p = Path(p)
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["month"]) if "month" in pd.read_csv(p, nrows=0).columns else pd.read_csv(p)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    allp = pd.concat(dfs, ignore_index=True)
    return allp


def plot_company(company_id: int, company_csv: Path, predictions_paths, out_path: Path):
    company_df = load_company_csv(company_csv)
    if company_df is None:
        raise FileNotFoundError(f"Company CSV not found: {company_csv}")

    preds_all = load_predictions(predictions_paths)
    if preds_all.empty:
        raise FileNotFoundError(f"No predictions found in: {predictions_paths}")

    # filter predictions for company
    preds = preds_all[preds_all['company'].astype(str) == str(company_id)].copy()
    if preds.empty:
        raise ValueError(f"No predictions for company {company_id} in provided prediction files")

    # normalize column names
    preds['month'] = pd.to_datetime(preds['month'])
    company_df = company_df.sort_values('Month')
    preds = preds.sort_values('month')

    # merge on nearest month if needed (use exact month join)
    merged = pd.merge(company_df, preds, left_on='Month', right_on='month', how='inner')
    if merged.empty:
        # try aligning on month strings
        merged = pd.merge(company_df, preds, left_on=company_df['Month'].dt.strftime('%Y-%m-%d'), right_on=preds['month'].dt.strftime('%Y-%m-%d'), how='inner')

    # choose actual return column
    if 'monthly_gross_return' in merged.columns:
        actual_col = 'monthly_gross_return'
    elif 'actual' in merged.columns:
        actual_col = 'actual'
    else:
        # attempt to find a numeric column that looks like return
        candidates = [c for c in merged.columns if merged[c].dtype.kind in 'fi' and c not in ('company', 'co_code')]
        actual_col = candidates[0] if candidates else None

    if actual_col is None:
        raise ValueError('Could not identify actual return column in company CSV')

    # model columns we expect in predictions
    model_cols = [c for c in preds.columns if c.startswith('catboost')]
    if not model_cols:
        # fallback: use all numeric prediction columns except keys
        model_cols = [c for c in preds.columns if preds[c].dtype.kind in 'fi' and c not in ('company','month')]

    # plotting
    # prefer seaborn style if available, otherwise fallback to default
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn')
        except Exception:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14,5))

    ax.plot(merged['Month'], merged[actual_col], label='Actual', color='black', linewidth=2)

    colors = plt.cm.tab10.colors
    for i, col in enumerate(model_cols):
        if col in merged.columns:
            ax.plot(merged['Month'], merged[col], label=col, color=colors[i % len(colors)], alpha=0.9)

    ax.set_title(f'Company {company_id} - Actual vs Model Predictions')
    ax.set_xlabel('Month')
    ax.set_ylabel('Return')
    ax.legend(loc='upper left', fontsize='small')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.autofmt_xdate()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'Saved overlay plot to {out_path}')

    # return merged data and model columns and full predictions for further per-model plotting
    return merged, model_cols, actual_col, preds_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--company', type=int, required=True)
    p.add_argument('--company-csv', type=str, default='holdout_set/{id}.csv')
    p.add_argument('--predictions', nargs='+', default=['results/rmse/test/predictions.csv', 'results/rmse/holdout/predictions.csv'])
    p.add_argument('--out', type=str, default='results/diagnostics/company_{id}_overlay.png')
    p.add_argument('--per-model', action='store_true', help='Also create a separate overlay for each model')
    p.add_argument('--clip-percentile', type=float, default=None, help='Clip model y-range to this central percentile (e.g. 99.5) to remove extreme outliers')
    args = p.parse_args()

    company_csv = Path(str(args.company_csv).format(id=args.company))
    out_path = Path(str(args.out).format(id=args.company))

    merged, model_cols, actual_col, preds_all = plot_company(args.company, company_csv, args.predictions, out_path)

    if args.per_model:
        per_dir = out_path.parent / f"company_{args.company}_per_model"
        per_dir.mkdir(parents=True, exist_ok=True)

        # compute mean RMSE across all companies for each model (global mean)
        mean_rmses = {}
        for col in model_cols:
            if col in preds_all.columns and 'actual' in preds_all.columns:
                dif = preds_all[col].astype(float) - preds_all['actual'].astype(float)
                mean_rmses[col] = np.sqrt(np.nanmean(dif ** 2))
            else:
                mean_rmses[col] = np.nan

        # save a global mean-rmse bar chart
        fig_bar, ax_bar = plt.subplots(figsize=(6,4))
        keys = list(mean_rmses.keys())
        vals = [mean_rmses[k] for k in keys]
        ax_bar.bar(range(len(keys)), vals, color=plt.cm.tab10.colors[:len(keys)])
        ax_bar.set_xticks(range(len(keys)))
        ax_bar.set_xticklabels(keys, rotation=45, ha='right', fontsize='small')
        ax_bar.set_ylabel('Mean RMSE')
        ax_bar.set_title('Mean RMSE by Architecture (global)')
        bar_path = per_dir / f'company_{args.company}_mean_rmse_by_arch.png'
        fig_bar.tight_layout()
        fig_bar.savefig(bar_path, dpi=150)
        plt.close(fig_bar)
        print(f'Saved mean RMSE bar chart: {bar_path}')

        # now per-model combined figures (timeseries top, scatter bottom-left, mean-rmse bar bottom-right)
        for col in model_cols:
            if col not in merged.columns:
                continue
            fig = plt.figure(figsize=(14,6))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])
            ax_ts = fig.add_subplot(gs[0, :])
            ax_sc = fig.add_subplot(gs[1, 0])
            ax_br = fig.add_subplot(gs[1, 1])

            # timeseries
            ax_ts.plot(merged['Month'], merged[actual_col], label='Actual', color='black', linewidth=2)
            ax_ts.plot(merged['Month'], merged[col], label=col, color='tab:blue', alpha=0.9)
            ax_ts.set_title(f'Company {args.company} - Actual vs {col}')
            ax_ts.set_xlabel('Month')
            ax_ts.set_ylabel('Return')
            ax_ts.legend()

            # scatter predicted vs actual
            ax_sc.scatter(merged[actual_col], merged[col], s=12, alpha=0.6)
            lo_line = min(merged[actual_col].min(), merged[col].min())
            hi_line = max(merged[actual_col].max(), merged[col].max())
            ax_sc.plot([lo_line, hi_line], [lo_line, hi_line], 'r--')
            ax_sc.set_xlabel('Actual')
            ax_sc.set_ylabel('Predicted')
            comp_rmse = np.sqrt(np.nanmean((merged[col].astype(float) - merged[actual_col].astype(float)) ** 2))
            ax_sc.set_title(f'Scatter (company RMSE={comp_rmse:.6f})')

            # bar: mean RMSEs (global)
            keys = list(mean_rmses.keys())
            vals = [mean_rmses[k] for k in keys]
            ax_br.barh(range(len(keys)), vals, color=plt.cm.tab10.colors[:len(keys)])
            ax_br.set_yticks(range(len(keys)))
            ax_br.set_yticklabels(keys, fontsize='small')
            ax_br.invert_yaxis()
            ax_br.set_xlabel('Mean RMSE')
            ax_br.set_title('Global mean RMSE')

            if args.clip_percentile:
                lo = np.percentile(merged[col].dropna().values, (100-args.clip_percentile)/2)
                hi = np.percentile(merged[col].dropna().values, 100-(100-args.clip_percentile)/2)
                lo = min(lo, merged[actual_col].min())
                hi = max(hi, merged[actual_col].max())
                ax_ts.set_ylim(lo, hi)

            fpath = per_dir / f'company_{args.company}_{col}_combined.png'
            fig.autofmt_xdate()
            fig.savefig(fpath, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f'Saved combined figure: {fpath}')


if __name__ == '__main__':
    main()
