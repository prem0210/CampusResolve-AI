from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/complaints_master.csv")

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


def validate_dataset() -> None:
    df = pd.read_csv(DATA_PATH)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    assert not missing_columns, f"Missing columns: {missing_columns}"
    assert df["complaint_id"].is_unique, "complaint_id values must be unique"
    assert df["complaint_text"].notna().all(), "complaint_text cannot contain null values"
    assert set(df["language"]).issubset({"en", "ta", "ta_en"}), "Unexpected language value"
    assert set(df["priority"]).issubset({"Low", "Medium", "High", "Critical"}), "Unexpected priority"
    assert set(df["is_duplicate"]).issubset({0, 1}), "is_duplicate must contain only 0 or 1"
    assert (df["resolution_time_hours"] > 0).all(), "Resolution time must be positive"

    duplicate_rows = df[df["is_duplicate"] == 1]
    assert duplicate_rows["duplicate_of_id"].notna().all(), (
        "Duplicate records must reference an original complaint ID"
    )

    print(f"Dataset validation passed: {len(df)} records, {df['category'].nunique()} categories.")
    print("\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print("\nLanguage distribution:")
    print(df["language"].value_counts().to_string())


if __name__ == "__main__":
    validate_dataset()