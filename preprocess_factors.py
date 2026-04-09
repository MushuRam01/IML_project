"""Preprocess factor_data.csv for CatBoost + residual NN modeling.

The script follows the cleaning notes in clean_data.md:
- sort by co_code and Month
- parse Month to datetime
- keep missing values and add missingness indicators
- coerce numeric columns to numeric types and categorical labels to strings
- engineer leak-safe lag and rolling return features
- create time-based or company-level train/validation/test splits (configurable)

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
import math
import random
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

RAW_RETURN_COLUMNS = [
    "lag_ret",
    "log_ret",
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
TRAIN_SHARE = 0.8
VALIDATION_SHARE = 0.1
TEST_SHARE = 0.1
VALIDATION_MIN_HISTORY = 6
DEFAULT_RANDOM_SEED = 13
TIME_SPLIT_TRAIN_END = "2019-09"
TIME_SPLIT_VALIDATION_END = "2023-09"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and feature-engineer factor_data.csv for modeling with time-based or company-level splits."
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
        "--split-mode",
        choices=("time", "company"),
        default="time",
        help="Split strategy: time-based (default) or company-based.",
    )
    parser.add_argument(
        "--train-end-month",
        default=TIME_SPLIT_TRAIN_END,
        help="Last month for training split (YYYY-MM) when using time splits.",
    )
    parser.add_argument(
        "--validation-end-month",
        default=TIME_SPLIT_VALIDATION_END,
        help="Last month for validation split (YYYY-MM) when using time splits.",
    )
    parser.add_argument(
        "--train-share",
        type=float,
        default=TRAIN_SHARE,
        help="Share of companies assigned to training.",
    )
    parser.add_argument(
        "--validation-share",
        type=float,
        default=VALIDATION_SHARE,
        help="Share of companies assigned to validation.",
    )
    parser.add_argument(
        "--test-share",
        type=float,
        default=TEST_SHARE,
        help="Share of companies assigned to test.",
    )
    parser.add_argument(
        "--split-date-source",
        choices=("month", "corrected"),
        default="month",
        help="Which date source to use when computing company history.",
    )
    parser.add_argument(
        "--validation-min-history",
        type=int,
        default=VALIDATION_MIN_HISTORY,
        help="Minimum months of history preferred for validation companies.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed for randomized company splits.",
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


def drop_raw_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [column for column in RAW_RETURN_COLUMNS if column in df.columns]
    if drop_cols:
        LOGGER.info("Dropping raw return columns to avoid leakage: %s", ", ".join(drop_cols))
        df = df.drop(columns=drop_cols)
    return df


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


def parse_month_period(value: str, label: str) -> pd.Period:
    try:
        return pd.Period(value, freq="M")
    except Exception as exc:
        raise ValueError(f"{label} must be YYYY-MM (got {value!r})") from exc


def assign_time_splits(
    df: pd.DataFrame,
    split_date_source: str,
    train_end_month: str,
    validation_end_month: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()

    train_end = parse_month_period(train_end_month, "--train-end-month")
    validation_end = parse_month_period(validation_end_month, "--validation-end-month")
    if train_end >= validation_end:
        raise ValueError("--train-end-month must be earlier than --validation-end-month")

    split_date = choose_split_date(df, split_date_source)
    split_month = split_date.dt.to_period("M")
    if split_month.isna().any():
        raise ValueError("Split assignment failed: missing split dates detected")

    df["split_group"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[split_month <= train_end, "split_group"] = "train"
    df.loc[(split_month > train_end) & (split_month <= validation_end), "split_group"] = "validation"
    df.loc[split_month > validation_end, "split_group"] = "test"

    valid_splits = {"train", "validation", "test"}
    if df["split_group"].isna().any():
        raise ValueError("Split assignment failed: missing split_group values")
    if not set(df["split_group"].unique()).issubset(valid_splits):
        raise ValueError("Split assignment failed: unexpected split_group values")

    company_key, missing_company_rows = build_company_key(df)
    if missing_company_rows:
        LOGGER.warning(
            "Found %s rows with missing %s; grouping them as a single company for splits.",
            missing_company_rows,
            ID_COLUMN,
        )

    history_counts = compute_company_history(df, company_key, split_date_source)
    return df, company_key, history_counts


def assign_splits(
    df: pd.DataFrame,
    train_share: float,
    validation_share: float,
    test_share: float,
    split_date_source: str,
    validation_min_history: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    validate_split_shares(train_share, validation_share, test_share)

    if validation_min_history < 0:
        raise ValueError("--validation-min-history must be non-negative")

    company_key, missing_company_rows = build_company_key(df)
    if missing_company_rows:
        LOGGER.warning(
            "Found %s rows with missing %s; grouping them as a single company for splits.",
            missing_company_rows,
            ID_COLUMN,
        )

    history_counts = compute_company_history(df, company_key, split_date_source)
    total_companies = int(history_counts.shape[0])
    train_count, validation_count, test_count = compute_split_counts(
        total_companies,
        train_share,
        validation_share,
        test_share,
    )

    rng = random.Random(random_seed)

    eligible = history_counts[history_counts >= validation_min_history].index.tolist()
    ineligible = history_counts[history_counts < validation_min_history].index.tolist()
    rng.shuffle(eligible)
    rng.shuffle(ineligible)

    if validation_count <= len(eligible):
        validation_companies = eligible[:validation_count]
        remaining = eligible[validation_count:] + ineligible
    else:
        needed = validation_count - len(eligible)
        validation_companies = eligible + ineligible[:needed]
        remaining = ineligible[needed:]

    rng.shuffle(remaining)
    train_companies = remaining[:train_count]
    test_companies = remaining[train_count:]

    company_split = {
        **{company: "train" for company in train_companies},
        **{company: "validation" for company in validation_companies},
        **{company: "test" for company in test_companies},
    }

    df["split_group"] = company_key.map(company_split)
    verify_split_integrity(df, company_key, total_companies)

    return df, company_key, history_counts


def validate_split_shares(train_share: float, validation_share: float, test_share: float) -> None:
    for share, label in (
        (train_share, "train"),
        (validation_share, "validation"),
        (test_share, "test"),
    ):
        if share <= 0:
            raise ValueError(f"{label} share must be greater than 0")
    total = train_share + validation_share + test_share
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train/validation/test shares must sum to 1.0")


def compute_split_counts(
    total_companies: int,
    train_share: float,
    validation_share: float,
    test_share: float,
) -> tuple[int, int, int]:
    if total_companies <= 0:
        raise ValueError("No companies available for splitting")

    raw_counts = {
        "train": total_companies * train_share,
        "validation": total_companies * validation_share,
        "test": total_companies * test_share,
    }
    counts = {name: int(math.floor(count)) for name, count in raw_counts.items()}
    remainder = total_companies - sum(counts.values())

    if remainder:
        fractions = sorted(
            ((name, raw_counts[name] - counts[name]) for name in counts),
            key=lambda item: item[1],
            reverse=True,
        )
        for index in range(remainder):
            counts[fractions[index][0]] += 1

    return counts["train"], counts["validation"], counts["test"]


def build_company_key(df: pd.DataFrame) -> tuple[pd.Series, int]:
    if ID_COLUMN not in df.columns:
        raise KeyError(f"Required column missing: {ID_COLUMN}")

    company_key = df[ID_COLUMN].astype("string")
    missing_count = int(company_key.isna().sum())
    if missing_count:
        company_key = company_key.fillna("__missing_co_code__")
    return company_key, missing_count


def compute_company_history(
    df: pd.DataFrame,
    company_key: pd.Series,
    split_date_source: str,
) -> pd.Series:
    split_date = choose_split_date(df, split_date_source)
    split_month = split_date.dt.to_period("M")
    return split_month.groupby(company_key).nunique(dropna=True)


def verify_split_integrity(
    df: pd.DataFrame,
    company_key: pd.Series,
    expected_company_count: int,
) -> None:
    valid_splits = {"train", "validation", "test"}
    if df["split_group"].isna().any():
        raise ValueError("Split assignment failed: missing split_group values")
    if not set(df["split_group"].unique()).issubset(valid_splits):
        raise ValueError("Split assignment failed: unexpected split_group values")

    company_split_counts = df.groupby(company_key)["split_group"].nunique(dropna=False)
    if (company_split_counts != 1).any():
        raise ValueError("Split assignment failed: company appears in multiple splits")

    if int(company_split_counts.shape[0]) != expected_company_count:
        raise ValueError("Split assignment failed: company count mismatch")


def build_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        TARGET_COLUMN,
        ID_COLUMN,
        RAW_DATE_COLUMN,
        "split_group",
    }
    excluded.update({column for column in df.columns if column.startswith("is_missing_")})
    return [column for column in df.columns if column not in excluded]


def write_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    metadata_extra: dict[str, object] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected; aborting output write")

    output_paths = {
        "processed_full": output_dir / "processed_full.csv",
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
        "test_targets": output_dir / "test_targets.csv",
        "metadata": output_dir / "metadata.json",
    }

    test_df = df[df["split_group"] == "test"].copy()
    leaky_columns = [TARGET_COLUMN, *RAW_RETURN_COLUMNS]
    test_truth_columns = [
        column
        for column in [ID_COLUMN, RAW_DATE_COLUMN, *leaky_columns]
        if column in test_df.columns
    ]
    test_truth = test_df[test_truth_columns].copy()
    for column in leaky_columns:
        if column in test_df.columns:
            test_df[column] = math.nan
            indicator_col = f"is_missing_{column}"
            if indicator_col in test_df.columns:
                test_df[indicator_col] = test_df[column].isna().astype("int8")

    safe_write_csv(df, output_paths["processed_full"])
    safe_write_csv(df[df["split_group"] == "train"], output_paths["train"])
    safe_write_csv(df[df["split_group"] == "validation"], output_paths["validation"])
    safe_write_csv(test_df, output_paths["test"])
    safe_write_csv(test_truth, output_paths["test_targets"])

    metadata = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "split_counts": df["split_group"].value_counts(dropna=False).to_dict(),
        "categorical_columns": [column for column in CATEGORICAL_COLUMNS if column in df.columns],
        "feature_columns": build_model_feature_columns(df),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    safe_write_text(json.dumps(metadata, indent=2, default=str), output_paths["metadata"])

    return output_paths


def safe_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise IOError(f"Failed to write CSV output: {output_path}")
    temp_path.replace(output_path)


def safe_write_text(content: str, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise IOError(f"Failed to write output file: {output_path}")
    temp_path.replace(output_path)


def log_summary(
    df: pd.DataFrame,
    indicator_columns: Iterable[str],
    company_key: pd.Series,
    history_counts: pd.Series,
    validation_min_history: int,
    split_mode: str,
) -> None:
    indicator_list = list(indicator_columns)
    LOGGER.info("Split mode: %s", split_mode)

    if split_mode == "company":
        company_split = df.groupby(company_key)["split_group"].first()
        company_split_counts = company_split.value_counts()
        validation_history = history_counts[company_split == "validation"]
        validation_with_history = int((validation_history >= validation_min_history).sum())
        validation_total = int(validation_history.shape[0])
    else:
        frame = df.assign(_company_key=company_key)
        company_split_counts = frame.groupby("split_group")["_company_key"].nunique()
        company_split_nunique = frame.groupby("_company_key")["split_group"].nunique(dropna=False)
        multi_split_companies = int((company_split_nunique > 1).sum())
        validation_companies = frame.loc[frame["split_group"] == "validation", "_company_key"].unique()
        validation_history = history_counts.loc[history_counts.index.isin(validation_companies)]
        validation_with_history = int((validation_history >= validation_min_history).sum())
        validation_total = int(validation_history.shape[0])
        LOGGER.info("Companies in multiple splits: %s", multi_split_companies)

    LOGGER.info("Rows: %s", len(df))
    LOGGER.info("Columns: %s", len(df.columns))
    LOGGER.info("Split counts (rows):\n%s", df["split_group"].value_counts(dropna=False).to_string())
    LOGGER.info("Split counts (companies):\n%s", company_split_counts.to_string())
    LOGGER.info(
        "Validation companies with >=%s months history: %s/%s",
        validation_min_history,
        validation_with_history,
        validation_total,
    )
    LOGGER.info("Missing indicators added: %s", len(indicator_list))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    df = read_csv(args.input)
    df = normalize_columns(df)
    df = coerce_types(df)
    df = drop_raw_return_columns(df)

    if df[RAW_DATE_COLUMN].isna().any():
        bad_rows = int(df[RAW_DATE_COLUMN].isna().sum())
        LOGGER.warning("Dropping %s rows with invalid Month values.", bad_rows)
        df = df.dropna(subset=[RAW_DATE_COLUMN]).copy()

    df = engineer_time_features(df)
    df, indicator_columns = add_missing_indicators(df)
    if args.split_mode == "time":
        df, company_key, history_counts = assign_time_splits(
            df,
            split_date_source=args.split_date_source,
            train_end_month=args.train_end_month,
            validation_end_month=args.validation_end_month,
        )
    else:
        df, company_key, history_counts = assign_splits(
            df,
            train_share=args.train_share,
            validation_share=args.validation_share,
            test_share=args.test_share,
            split_date_source=args.split_date_source,
            validation_min_history=args.validation_min_history,
            random_seed=args.random_seed,
        )

    df = df.sort_values([ID_COLUMN, RAW_DATE_COLUMN], kind="mergesort").reset_index(drop=True)

    log_summary(
        df,
        indicator_columns,
        company_key,
        history_counts,
        args.validation_min_history,
        args.split_mode,
    )
    metadata_extra = {
        "split_mode": args.split_mode,
        "split_date_source": args.split_date_source,
    }
    if args.split_mode == "time":
        metadata_extra["time_split"] = {
            "train_end_month": args.train_end_month,
            "validation_end_month": args.validation_end_month,
        }
    else:
        metadata_extra["company_split"] = {
            "train_share": args.train_share,
            "validation_share": args.validation_share,
            "test_share": args.test_share,
            "validation_min_history": args.validation_min_history,
        }
    output_paths = write_outputs(df, args.output_dir, metadata_extra=metadata_extra)

    LOGGER.info("Wrote processed data to %s", output_paths["processed_full"])
    LOGGER.info("Wrote train/validation/test splits to %s", args.output_dir)


if __name__ == "__main__":
    main()
