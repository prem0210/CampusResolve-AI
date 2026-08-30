from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

INPUT_PATH = Path("data/processed/complaints_master_synthetic.csv")
OUTPUT_DIR = Path("data/processed")

REQUIRED_COLUMNS = {
    "complaint_id",
    "created_at",
    "language",
    "complaint_text",
    "normalized_text",
    "category",
    "department",
    "priority",
    "location_type",
    "specific_location",
    "affected_population",
    "safety_flag",
    "repeat_count",
    "is_duplicate",
    "duplicate_of_id",
    "resolution_time_hours",
    "status",
    "resolution_notes",
}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()

    df["complaint_text"] = df["complaint_text"].fillna("").astype(str).str.strip()
    df["normalized_text"] = df["normalized_text"].fillna("").astype(str).str.strip()
    df["resolution_notes"] = df["resolution_notes"].fillna("").astype(str).str.strip()
    df["duplicate_of_id"] = df["duplicate_of_id"].fillna("").astype(str).str.strip()

    df = df[df["complaint_text"].ne("")].copy()
    df = df.drop_duplicates(subset=["complaint_id"]).copy()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).copy()

    numeric_columns = [
        "affected_population",
        "safety_flag",
        "repeat_count",
        "is_duplicate",
        "resolution_time_hours",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()
    df["affected_population"] = df["affected_population"].clip(lower=1).astype(int)
    df["safety_flag"] = df["safety_flag"].astype(int)
    df["repeat_count"] = df["repeat_count"].clip(lower=0).astype(int)
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    df["resolution_time_hours"] = df["resolution_time_hours"].clip(lower=0.5)

    valid_languages = {"en", "ta", "ta_en"}
    valid_priorities = {"Low", "Medium", "High", "Critical"}

    df = df[df["language"].isin(valid_languages)].copy()
    df = df[df["priority"].isin(valid_priorities)].copy()

    return df.reset_index(drop=True)


def create_group_id(df: pd.DataFrame) -> pd.Series:
    return df["duplicate_of_id"].where(
        df["duplicate_of_id"].ne(""),
        df["complaint_id"],
    )


def grouped_split(
    df: pd.DataFrame,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_indices, test_indices = next(splitter.split(df, groups=groups))

    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True),
    )


def validate_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    split_groups = {
        "train": set(create_group_id(train_df)),
        "validation": set(create_group_id(validation_df)),
        "test": set(create_group_id(test_df)),
    }

    assert split_groups["train"].isdisjoint(split_groups["validation"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["validation"].isdisjoint(split_groups["test"])

    all_ids = set(train_df["complaint_id"])
    all_ids.update(validation_df["complaint_id"])
    all_ids.update(test_df["complaint_id"])

    assert len(all_ids) == len(train_df) + len(validation_df) + len(test_df)
    assert all(split["category"].nunique() == 10 for split in [train_df, validation_df, test_df])


def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")


def print_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    print("\nSplit sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    for split_name, split_df in {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df,
    }.items():
        print(f"\n{split_name} category distribution:")
        print(split_df["category"].value_counts().sort_index().to_string())

        print(f"\n{split_name} priority distribution:")
        print(split_df["priority"].value_counts().reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0,
        ).to_string())

        print(f"\n{split_name} duplicate count: {int(split_df['is_duplicate'].sum())}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}. Run generate_dataset.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    df = clean_dataset(df)
    groups = create_group_id(df)

    train_validation_df, test_df = grouped_split(
        df=df,
        groups=groups,
        test_size=0.20,
        random_state=42,
    )

    train_validation_groups = create_group_id(train_validation_df)
    train_df, validation_df = grouped_split(
        df=train_validation_df,
        groups=train_validation_groups,
        test_size=0.125,
        random_state=42,
    )

    validate_splits(train_df, validation_df, test_df)
    save_splits(train_df, validation_df, test_df)
    print_summary(train_df, validation_df, test_df)


if __name__ == "__main__":
    main()