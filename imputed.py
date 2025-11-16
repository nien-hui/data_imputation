import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_object_dtype


REMASKER_DIR_CANDIDATES = [Path("remasker_main"), Path("remasker-main")]
for candidate in REMASKER_DIR_CANDIDATES:
    if candidate.exists():
        candidate_path = str(candidate.resolve())
        if candidate_path not in sys.path:
            sys.path.insert(0, candidate_path)
        break
else:
    raise FileNotFoundError(
        "Unable to locate the remasker_main directory. Update REMASKER_DIR_CANDIDATES to match your project layout."
    )

from remasker_impute import ReMasker


ID_COLUMN = "SK_ID_CURR"
RARE_CATEGORY_THRESHOLD = 0.01
FLOAT_DTYPE = "float32"

COMBINED_DATA_PATH = Path("combined.csv")
TRAIN_DATA_PATH = Path("home-credit-default-risk") / "application_train.csv"
TEST_DATA_PATH = Path("home-credit-default-risk") / "application_test.csv"
CHECKPOINT_PATH = Path("checkpoints") / ""
IMPUTED_TRAIN_OUTPUT = Path("imputed_train.csv")
IMPUTED_TEST_OUTPUT = Path("imputed_test.csv")

REMASKER_BATCH_SIZE = 256
REMASKER_EPOCHS_FIT = 100
REMASKER_EPOCHS_LOAD = 10


def ensure_integer_ids(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    df = df.copy()
    df[id_column] = pd.to_numeric(df[id_column], errors="coerce")
    missing = df[id_column].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows in the combined data set do not contain a valid {id_column}.")
    df[id_column] = df[id_column].astype("int64")
    return df


def encode_categoricals(df: pd.DataFrame, id_column: str, threshold: float) -> pd.DataFrame:
    processed = df.copy()
    for col in df.columns:
        if col == id_column:
            continue

        series = df[col]
        value_counts = series.value_counts(normalize=True)
        valid_categories = value_counts[value_counts >= threshold].index.tolist()
        total_categories = len(value_counts)

        if total_categories == 2 and len(valid_categories) < 2:
            print(f"Dropping binary column '{col}' because its minority category < {threshold:.2%}.")
            processed = processed.drop(columns=[col])
            continue

        if is_object_dtype(series):
            if valid_categories:
                dummies = pd.get_dummies(series, dummy_na=False)
                valid_existing = [cat for cat in valid_categories if cat in dummies.columns]
                if valid_existing:
                    dummies = dummies[valid_existing]
                    dummies.columns = [f"{col}_{cat}" for cat in valid_existing]
                    processed = pd.concat([processed, dummies], axis=1)
            processed = processed.drop(columns=[col])

    return processed


def prepare_numeric_data(df: pd.DataFrame, id_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    processed = df.copy()
    bool_columns = [col for col in processed.select_dtypes(include="bool").columns if col != id_column]
    if bool_columns:
        print(f"Converting {len(bool_columns)} boolean columns to {FLOAT_DTYPE}.")
        processed[bool_columns] = processed[bool_columns].astype(FLOAT_DTYPE)

    numeric_columns = [col for col in processed.select_dtypes(include="number").columns if col != id_column]
    if not numeric_columns:
        raise ValueError("No numeric columns available for imputation.")

    numeric_data = processed[numeric_columns].astype(FLOAT_DTYPE)
    return processed, numeric_data, numeric_columns


def build_remasker(max_epochs: int, batch_size: int) -> ReMasker:
    argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0], "--max_epochs", str(max_epochs), "--batch_size", str(batch_size)]
    try:
        return ReMasker()
    finally:
        sys.argv = argv_backup


def impute_numeric(
    numeric_data: pd.DataFrame,
    numeric_columns: List[str],
    checkpoint_path: Path,
) -> pd.DataFrame:
    use_checkpoint = checkpoint_path.exists()
    max_epochs = REMASKER_EPOCHS_LOAD if use_checkpoint else REMASKER_EPOCHS_FIT
    remasker = build_remasker(max_epochs=max_epochs, batch_size=REMASKER_BATCH_SIZE)

    if use_checkpoint:
        print(f"Loading checkpoint from {checkpoint_path} for imputation.")
        imputed_array = remasker.laod_param_fit_transform(numeric_data, str(checkpoint_path))
    else:
        print(f"Checkpoint {checkpoint_path} not found. Running fit_transform to train from scratch.")
        imputed_array = remasker.fit_transform(numeric_data)

    return pd.DataFrame(imputed_array, columns=numeric_columns, index=numeric_data.index)


def split_imputed_datasets(
    imputed_numeric: pd.DataFrame,
    combined_df: pd.DataFrame,
    train_path: Path,
    test_path: Path,
    id_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_count = len(train_df)
    test_count = len(test_df)
    total_count = train_count + test_count
    if total_count != len(imputed_numeric):
        raise ValueError(
            f"Combined length mismatch: expected {total_count} rows from train/test but got {len(imputed_numeric)}."
        )

    train_ids = combined_df.iloc[:train_count][id_column].astype("int64").to_numpy()
    test_ids = combined_df.iloc[train_count:][id_column].astype("int64").to_numpy()

    imputed_train = imputed_numeric.iloc[:train_count].copy()
    imputed_test = imputed_numeric.iloc[train_count:].copy()
    imputed_train.insert(0, id_column, train_ids)
    imputed_test.insert(0, id_column, test_ids)

    return imputed_train, imputed_test, train_df


def attach_target_column(imputed_train: pd.DataFrame, train_df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    if "TARGET" not in train_df.columns:
        raise ValueError("TARGET column not found in training data.")

    target_df = train_df[[id_column, "TARGET"]].copy()
    target_df[id_column] = pd.to_numeric(target_df[id_column], errors="coerce")
    target_df = target_df.dropna(subset=[id_column, "TARGET"])
    target_df[id_column] = target_df[id_column].astype("int64")

    merged = imputed_train.merge(target_df, on=id_column, how="left")
    missing_targets = merged["TARGET"].isna().sum()
    if missing_targets:
        raise ValueError(f"{missing_targets} training rows are missing TARGET labels after merging.")

    return merged


def main() -> None:
    combined_data = ensure_integer_ids(pd.read_csv(COMBINED_DATA_PATH, low_memory=False), ID_COLUMN)
    encoded_data = encode_categoricals(combined_data, ID_COLUMN, RARE_CATEGORY_THRESHOLD)
    processed_data, numeric_data, numeric_columns = prepare_numeric_data(encoded_data, ID_COLUMN)
    print(f"Prepared {len(numeric_columns)} numeric columns for ReMasker.")

    imputed_numeric = impute_numeric(numeric_data, numeric_columns, CHECKPOINT_PATH)
    print(f"Completed imputation for {imputed_numeric.shape[0]} rows.")

    imputed_train, imputed_test, train_df = split_imputed_datasets(
        imputed_numeric, processed_data, TRAIN_DATA_PATH, TEST_DATA_PATH, ID_COLUMN
    )
    print(f"Imputed train shape: {imputed_train.shape}; test shape: {imputed_test.shape}.")

    imputed_train_with_target = attach_target_column(imputed_train, train_df, ID_COLUMN)
    imputed_train_with_target.to_csv(IMPUTED_TRAIN_OUTPUT, index=False)
    print(f"Wrote imputed training data with TARGET to {IMPUTED_TRAIN_OUTPUT}.")

    imputed_test.to_csv(IMPUTED_TEST_OUTPUT, index=False)
    print(f"Wrote imputed test data to {IMPUTED_TEST_OUTPUT}.")


if __name__ == "__main__":
    main()
