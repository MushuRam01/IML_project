# ============================================================
# evaluation/generate_dashboards.py
# ============================================================
# Generates dashboards for Test and Holdout splits using the
# consolidated predictions produced under results/rmse/*.
# ============================================================

import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = BASE_DIR / "results" / "rmse"

# ============================================================
# Config
# ============================================================

MODEL_CONFIG = {
    "catboost": {"label": "CatBoost", "color": "#1f77b4", "linestyle": "--"},
    "catboost_hmm": {"label": "CatBoost + HMM", "color": "#2ca02c", "linestyle": "-."},
    "catboost_transformer": {"label": "CatBoost + Transformer", "color": "#ff7f0e", "linestyle": ":"},
    "catboost_mlp": {"label": "CatBoost + MLP", "color": "#8c564b", "linestyle": "-"},
    "catboost_cnn": {"label": "CatBoost + CNN", "color": "#9467bd", "linestyle": "-"},
    "catboost_ridge": {"label": "CatBoost + Ridge", "color": "#17becf", "linestyle": "-"},
    "catboost_elasticnet": {"label": "CatBoost + ElasticNet", "color": "#bcbd22", "linestyle": "-"},
    "catboost_residual": {"label": "CatBoost + Residual CatBoost", "color": "#e377c2", "linestyle": "-"},
}

# ============================================================
# Helpers
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_predictions(mode):
    split_dir = RESULTS_FOLDER / mode
    csv_path = split_dir / "predictions.csv"
    if not csv_path.exists():
        print(f"Missing predictions file: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df, split_dir


def build_company_rmse(df, model_cols):
    rows = []
    for company, group in df.groupby("company"):
        entry = {"company": company, "n_months": len(group)}
        for col in model_cols:
            entry[f"{col}_rmse"] = rmse(group["actual"].values, group[col].values)
        rows.append(entry)
    return pd.DataFrame(rows)


def build_summary(df, company_df, model_cols):
    rows = []
    for col in model_cols:
        rows.append({
            "model": col,
            "global_rmse": rmse(df["actual"].values, df[col].values),
            "min_company_rmse": company_df[f"{col}_rmse"].min(),
            "max_company_rmse": company_df[f"{col}_rmse"].max(),
            "mean_company_rmse": company_df[f"{col}_rmse"].mean(),
        })
    return pd.DataFrame(rows)


def plot_scatter_grid(df, summary_df, model_cols, output_path):
    n_models = len(model_cols)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    y = df["actual"].values
    min_val = float(np.nanmin(y))
    max_val = float(np.nanmax(y))

    for idx, col in enumerate(model_cols):
        ax = axes[idx]
        pred = df[col].values
        rmse_val = summary_df.loc[summary_df["model"] == col, "global_rmse"].values[0]
        label = MODEL_CONFIG[col]["label"]

        ax.scatter(y, pred, alpha=0.15, s=8, color=MODEL_CONFIG[col]["color"])
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_title(f"{label} (RMSE={rmse_val:.4f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_error_kde(df, model_cols, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in model_cols:
        err = df[col].values - df["actual"].values
        sns.kdeplot(err, ax=ax, label=MODEL_CONFIG[col]["label"], color=MODEL_CONFIG[col]["color"], fill=False)

    ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("Error Distribution (Residuals)")
    ax.set_xlabel("Predicted - Actual")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_stats(summary_df, mode, output_path):
    lines = [
        f"{mode.upper()} SET STATISTICS",
        "----------------------------------------",
    ]

    for _, row in summary_df.iterrows():
        label = MODEL_CONFIG[row["model"]]["label"]
        lines.append(
            f"{label}: RMSE {row['global_rmse']:.6f} | "
            f"Min {row['min_company_rmse']:.6f} | "
            f"Max {row['max_company_rmse']:.6f} | "
            f"Mean {row['mean_company_rmse']:.6f}"
        )

    text = "\n".join(lines)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.text(0.01, 0.99, text, fontsize=11, family="monospace", verticalalignment="top")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_timeline(df, company_id, model_cols, output_path, title):
    comp_df = df[df["company"] == company_id].sort_values("month")
    if comp_df.empty:
        return

    months = comp_df["month"].values
    actual = comp_df["actual"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(months, actual, label="Actual", color="black", linewidth=2)

    for col in model_cols:
        cfg = MODEL_CONFIG[col]
        ax.plot(months, comp_df[col].values, label=cfg["label"], color=cfg["color"], linestyle=cfg["linestyle"], alpha=0.9)

    ax.axhline(1.0, color="grey", linewidth=0.6, alpha=0.5)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

# ============================================================
# Dashboard Generation
# ============================================================

def generate_dashboard(mode):
    df, split_dir = load_predictions(mode)

    model_cols = [k for k in MODEL_CONFIG.keys() if k in df.columns]

    company_df = build_company_rmse(df, model_cols)
    company_df.to_csv(split_dir / "company_rmse.csv", index=False)

    summary_df = build_summary(df, company_df, model_cols)
    summary_df.to_csv(split_dir / "rmse_summary.csv", index=False)

    # Scatter grid
    plot_scatter_grid(df, summary_df, model_cols, split_dir / "scatter_grid.png")

    # Bell curves
    plot_error_kde(df, model_cols, split_dir / "error_kde.png")

    # Stats text
    plot_stats(summary_df, mode, split_dir / "stats.png")

    # Timeline plots
    company_df["avg_rmse"] = company_df[[f"{c}_rmse" for c in model_cols]].mean(axis=1)
    best_company = company_df.sort_values("avg_rmse").iloc[0]["company"]
    random_company = company_df.sample(1, random_state=42)["company"].values[0]

    plot_timeline(
        df,
        best_company,
        model_cols,
        split_dir / "timeline_best.png",
        f"Timeline: Company {best_company} ({mode.capitalize()} Set) (Best Fit)",
    )

    plot_timeline(
        df,
        random_company,
        model_cols,
        split_dir / "timeline_random.png",
        f"Timeline: Company {random_company} ({mode.capitalize()} Set) (Random Sample)",
    )

    print(f"Dashboards saved under {split_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "holdout", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ["test", "all"]:
        generate_dashboard("test")
    if args.mode in ["holdout", "all"]:
        generate_dashboard("holdout")
