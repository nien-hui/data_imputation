import os
from pathlib import Path
from typing import List

import pandas as pd


def _ensure_sk_id_curr_int(df: pd.DataFrame) -> pd.DataFrame:
    if "SK_ID_CURR" not in df.columns:
        raise KeyError("SK_ID_CURR not found in DataFrame")
    # Coerce to numeric then to int, handling float-like ids (e.g., 100001.0)
    df["SK_ID_CURR"] = pd.to_numeric(df["SK_ID_CURR"], errors="coerce")
    # Drop rows where key is missing after coercion
    df = df.dropna(subset=["SK_ID_CURR"]).copy()
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("int64")
    return df


def _clean_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop common auto-saved index columns from CSV exports
    drop_cols = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def load_applications(base_dir: Path) -> pd.DataFrame:
    train_path = base_dir / "home-credit-default-risk" / "application_train.csv"
    test_path = base_dir / "home-credit-default-risk" / "application_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    # Drop label from train if present
    if "TARGET" in train.columns:
        train = train.drop(columns=["TARGET"])  # keep only features

    # Ensure key type aligns
    train = _ensure_sk_id_curr_int(train)
    test = _ensure_sk_id_curr_int(test)

    # Align columns between train/test to avoid concat misalignment due to extras
    # Keep union of columns, filling missing with NaN
    common_cols = sorted(set(train.columns).union(test.columns))
    train = train.reindex(columns=common_cols)
    test = test.reindex(columns=common_cols)

    combined = pd.concat([train, test], ignore_index=True)
    return combined


def list_feature_files(transformed_dir: Path) -> List[Path]:
    if not transformed_dir.exists():
        return []
    return sorted(p for p in transformed_dir.glob("*.csv") if p.is_file())


def load_feature_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _clean_index_columns(df)
    df = _ensure_sk_id_curr_int(df)
    # Deduplicate on key defensively
    df = df.drop_duplicates(subset=["SK_ID_CURR"], keep="first")
    return df


def merge_features(base: pd.DataFrame, feature_files: List[Path]) -> pd.DataFrame:
    out = base
    for fpath in feature_files:
        stem = fpath.stem.lstrip("_") or fpath.stem
        feat = load_feature_frame(fpath)
        # Avoid carrying duplicate key columns from right df; merge handles this, but
        # suffix any overlapping non-key column names deterministically by file stem.
        overlap = [c for c in feat.columns if c != "SK_ID_CURR" and c in out.columns]
        suffix = f"_{stem}"
        out = out.merge(
            feat,
            on="SK_ID_CURR",
            how="left",
            suffixes=("", suffix),
        )
        # If there were overlaps, pandas would have suffixed columns from 'feat'.
        # Nothing else needed; keep deterministic order by merging in sorted file order.
        print(f"Merged features from {fpath.name} ({len(feat)} rows): overlap={len(overlap)}")
    return out


def main():
    base_dir = Path.cwd()
    transformed_dir = base_dir / "transformed_data"

    print("Loading applications (train+test) and dropping train label ...")
    combined = load_applications(base_dir)
    print(f"Combined shape after concat: {combined.shape}")

    print(f"Scanning feature files in {transformed_dir} ...")
    feature_files = list_feature_files(transformed_dir)
    print(f"Found {len(feature_files)} feature file(s)")

    if feature_files:
        combined = merge_features(combined, feature_files)
        print(f"Shape after merging features: {combined.shape}")
    else:
        print("No feature files found; skipping feature merge.")

    out_path = base_dir / "combined.csv"
    print(f"Writing combined dataset to {out_path} ...")
    # Index False to avoid writing numeric index column
    combined.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    # Allow pandas to display wide columns if the user runs interactively
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    main()
