"""Preprocess factor_data.csv for CatBoost + residual NN modeling.

The script follows the cleaning notes in clean_data.md:
- sort by co_code and Month
- parse Month to datetime
- keep missing values and add missingness indicators
- coerce numeric columns to numeric types and categorical labels to strings
- engineer leak-safe lag and rolling return features
- create time-based train/validation/test splits

Outputs are written to an output directory:
- processed_full.csv
- train.csv
- validation.csv
- test.csv
- metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise
    raise SystemExit(
        "Missing dependency: pandas. Install it with `python3 -m pip install pandas` "
        "and rerun the script."
    ) from exc


LOGGER = logging.getLogger("preprocess_factors")

TARGET_COLUMN = "monthly_gross_return"
ID_COLUMN = "co_code"
RAW_DATE_COLUMN = "Month"

CATEGORICAL_COLUMNS = [
    "Size_Label",
    "BM_Label",
    "OpProf_Label",
    "Inv_Label",
    "Mom_Label",
]

TIME_COLUMNS = [
    "Month",
    "Year",
    "Corrected_Year",
    "Corrected_Month",
    "Month_annual",
]

LEAK_SAFETY_LAGS = [1, 3, 6]
ROLLING_WINDOWS = [3, 6]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and feature-engineer factor_data.csv for modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "factor_data.csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "processed_data",
        help="Directory for processed outputs.",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2019,
        help="Last year included in the training split.",
    )
    parser.add_argument(
        "--validation-end-year",
        type=int,
        default=2021,
        help="Last year included in the validation split.",
    )
    parser.add_argument(
        "--split-date-source",
        choices=("month", "corrected"),
        default="month",
        help="Which date source to use for the train/validation/test split.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if RAW_DATE_COLUMN not in df.columns:
        raise KeyError(f"Required column missing: {RAW_DATE_COLUMN}")

    df[RAW_DATE_COLUMN] = pd.to_datetime(df[RAW_DATE_COLUMN], errors="coerce")

    if ID_COLUMN in df.columns:
        df[ID_COLUMN] = pd.to_numeric(df[ID_COLUMN], errors="coerce").astype("Int64")

    for column in ["Year", "Corrected_Year", "Corrected_Month", "Month_annual"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(float)

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
            )

    excluded = set([ID_COLUMN, TARGET_COLUMN, RAW_DATE_COLUMN, *CATEGORICAL_COLUMNS, *TIME_COLUMNS])
    numeric_candidates = [column for column in df.columns if column not in excluded]
    for column in numeric_candidates:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    return df


def add_missing_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    indicator_columns: list[str] = []
    for column in df.columns:
        if column in {ID_COLUMN, RAW_DATE_COLUMN}:
            continue
        if df[column].isna().any():
            indicator_name = f"is_missing_{column}"
            df[indicator_name] = df[column].isna().astype("int8")
            indicator_columns.append(indicator_name)
    return df, indicator_columns


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([ID_COLUMN, RAW_DATE_COLUMN], kind="mergesort").reset_index(drop=True)

    grouped = df.groupby(ID_COLUMN, sort=False)

    for lag in LEAK_SAFETY_LAGS:
        df[f"ret_lag_{lag}"] = grouped[TARGET_COLUMN].shift(lag)

    past_return = grouped[TARGET_COLUMN].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"ret_rolling_mean_{window}m"] = (
            past_return.groupby(df[ID_COLUMN]).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"ret_rolling_std_{window}m"] = (
            past_return.groupby(df[ID_COLUMN]).rolling(window=window, min_periods=2).std().reset_index(level=0, drop=True)
        )

    df["ret_rolling_count_3m"] = (
        past_return.groupby(df[ID_COLUMN]).rolling(window=3, min_periods=1).count().reset_index(level=0, drop=True)
    )

    if "log_ret" in df.columns:
        for lag in LEAK_SAFETY_LAGS:
            df[f"log_ret_lag_{lag}"] = grouped["log_ret"].shift(lag)

    df["month_index"] = df[RAW_DATE_COLUMN].dt.year * 12 + df[RAW_DATE_COLUMN].dt.month
    return df


def choose_split_date(df: pd.DataFrame, split_date_source: str) -> pd.Series:
    if split_date_source == "corrected" and {"Corrected_Year", "Corrected_Month"}.issubset(df.columns):
        split_date = pd.to_datetime(
            dict(year=df["Corrected_Year"], month=df["Corrected_Month"], day=1),
            errors="coerce",
        )
        fallback = df[RAW_DATE_COLUMN]
        return split_date.fillna(fallback)
    return df[RAW_DATE_COLUMN]


def assign_splits(
    df: pd.DataFrame,
    train_end_year: int,
    validation_end_year: int,
    split_date_source: str,
) -> pd.DataFrame:
    df = df.copy()
    split_date = choose_split_date(df, split_date_source)
    split_year = split_date.dt.year

    df["split_group"] = "test"
    df.loc[split_year <= train_end_year, "split_group"] = "train"
    df.loc[(split_year > train_end_year) & (split_year <= validation_end_year), "split_group"] = "validation"
    return df


def build_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        TARGET_COLUMN,
        ID_COLUMN,
        RAW_DATE_COLUMN,
        "split_group",
    }
    excluded.update({column for column in df.columns if column.startswith("is_missing_")})
    return [column for column in df.columns if column not in excluded]


def write_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "processed_full": output_dir / "processed_full.csv",
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
        "metadata": output_dir / "metadata.json",
    }

    df.to_csv(output_paths["processed_full"], index=False)
    df[df["split_group"] == "train"].to_csv(output_paths["train"], index=False)
    df[df["split_group"] == "validation"].to_csv(output_paths["validation"], index=False)
    df[df["split_group"] == "test"].to_csv(output_paths["test"], index=False)

    metadata = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "split_counts": df["split_group"].value_counts(dropna=False).to_dict(),
        "categorical_columns": [column for column in CATEGORICAL_COLUMNS if column in df.columns],
        "feature_columns": build_model_feature_columns(df),
    }
    output_paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return output_paths


def log_summary(df: pd.DataFrame, indicator_columns: Iterable[str]) -> None:
    indicator_list = list(indicator_columns)
    LOGGER.info("Rows: %s", len(df))
    LOGGER.info("Columns: %s", len(df.columns))
    LOGGER.info("Split counts:\n%s", df["split_group"].value_counts(dropna=False).to_string())
    LOGGER.info("Missing indicators added: %s", len(indicator_list))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.validation_end_year < args.train_end_year:
        raise ValueError(
            "--validation-end-year must be greater than or equal to --train-end-year"
        )

    df = read_csv(args.input)
    df = normalize_columns(df)
    df = coerce_types(df)

    if df[RAW_DATE_COLUMN].isna().any():
        bad_rows = int(df[RAW_DATE_COLUMN].isna().sum())
        LOGGER.warning("Dropping %s rows with invalid Month values.", bad_rows)
        df = df.dropna(subset=[RAW_DATE_COLUMN]).copy()

    df = engineer_time_features(df)
    df, indicator_columns = add_missing_indicators(df)
    df = assign_splits(
        df,
        train_end_year=args.train_end_year,
        validation_end_year=args.validation_end_year,
        split_date_source=args.split_date_source,
    )

    df = df.sort_values([ID_COLUMN, RAW_DATE_COLUMN], kind="mergesort").reset_index(drop=True)

    log_summary(df, indicator_columns)
    output_paths = write_outputs(df, args.output_dir)

    LOGGER.info("Wrote processed data to %s", output_paths["processed_full"])
    LOGGER.info("Wrote train/validation/test splits to %s", args.output_dir)


if __name__ == "__main__":
    main()
