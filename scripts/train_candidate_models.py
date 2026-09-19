from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
SCHEMA_VERSION = "1.0"

REQUIRED_COLUMNS = {
    "complaint_id",
    "complaint_reference",
    "complaint_text",
    "language",
    "location_type",
    "specific_location",
    "reported_affected_population",
    "verified_affected_population",
    "impact_verification_status",
    "final_category",
    "final_department_id",
    "final_priority",
    "actual_resolution_hours",
    "duplicate_decision",
    "complaint_status",
    "complaint_created_at",
    "feedback_reviewed_at",
    "feedback_reviewed_by_user_id",
}

MODEL_TARGETS = {
    "category": "final_category",
    "department": "final_department_id",
    "priority": "final_priority",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train candidate CampusResolve-AI classifiers from a governed "
            "A21 training-feedback export. This script never replaces active "
            "production artifacts."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to an A21 CSV training-feedback export.",
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the matching A21 JSON manifest.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/candidates"),
        help="Directory where candidate run folders will be created.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout fraction, strictly between 0 and 1.",
    )

    parser.add_argument(
        "--min-records",
        type=int,
        default=30,
        help="Minimum valid records needed before training starts.",
    )

    parser.add_argument(
        "--min-examples-per-class",
        type=int,
        default=3,
        help="Minimum examples per target class for each candidate model.",
    )

    return parser.parse_args()


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"PRECHECK FAILED: {message}")
    raise SystemExit(1)


def load_and_verify_manifest(
    csv_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if not csv_path.is_file():
        fail(f"CSV file does not exist: {csv_path}")

    if not manifest_path.is_file():
        fail(f"Manifest file does not exist: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"Manifest is not valid JSON: {error}")

    expected_hash = manifest.get("sha256")

    if not expected_hash:
        fail("Manifest does not contain sha256.")

    actual_hash = sha256_file(csv_path)

    if actual_hash != expected_hash:
        fail(
            "CSV SHA-256 does not match the manifest. "
            f"Expected {expected_hash}, received {actual_hash}."
        )

    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail(
            "Unsupported manifest schema version. "
            f"Expected {SCHEMA_VERSION}, received "
            f"{manifest.get('schema_version')}."
        )

    return manifest


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(csv_path)
    except Exception as error:
        fail(f"Could not read CSV: {error}")

    missing_columns = sorted(REQUIRED_COLUMNS - set(dataset.columns))

    if missing_columns:
        fail(
            "CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    if dataset.empty:
        fail("CSV contains no eligible training records.")

    dataset = dataset.copy()

    text_columns = [
        "complaint_text",
        "final_category",
        "final_priority",
        "impact_verification_status",
    ]

    for column in text_columns:
        dataset[column] = dataset[column].fillna("").astype(str).str.strip()

    dataset["final_department_id"] = pd.to_numeric(
        dataset["final_department_id"],
        errors="coerce",
    )

    valid_statuses = {"Verified", "Adjusted"}
    valid_complaint_statuses = {"Resolved", "Closed"}

    dataset = dataset[
        dataset["impact_verification_status"].isin(valid_statuses)
        & dataset["complaint_status"].isin(valid_complaint_statuses)
        & dataset["verified_affected_population"].notna()
        & dataset["final_category"].ne("")
        & dataset["final_priority"].ne("")
        & dataset["final_department_id"].notna()
        & dataset["complaint_text"].ne("")
    ].copy()

    if dataset.empty:
        fail(
            "No rows remain after governed-data validation. "
            "Ensure exported records have verified impact, final labels, "
            "and Resolved or Closed complaint status."
        )

    dataset["final_department_id"] = (
        dataset["final_department_id"].astype(int).astype(str)
    )

    return dataset


def validate_training_readiness(
    dataset: pd.DataFrame,
    min_records: int,
    min_examples_per_class: int,
) -> dict[str, dict[str, int]]:
    if len(dataset) < min_records:
        fail(
            f"Only {len(dataset)} valid records are available; "
            f"at least {min_records} are required."
        )

    class_counts: dict[str, dict[str, int]] = {}

    for model_name, target_column in MODEL_TARGETS.items():
        counts = dataset[target_column].value_counts()
        class_counts[model_name] = {
            str(label): int(count)
            for label, count in counts.to_dict().items()
        }

        if len(counts) < 2:
            fail(
                f"Target '{target_column}' has only one class. "
                "At least two classes are required."
            )

        too_small = counts[counts < min_examples_per_class]

        if not too_small.empty:
            details = ", ".join(
                f"{label}={count}"
                for label, count in too_small.to_dict().items()
            )
            fail(
                f"Target '{target_column}' has classes below the minimum "
                f"of {min_examples_per_class}: {details}."
            )

    return class_counts


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_and_evaluate_model(
    model_name: str,
    target_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    run_directory: Path,
) -> dict[str, Any]:
    pipeline = build_pipeline()

    pipeline.fit(
        train_data["complaint_text"],
        train_data[target_column],
    )

    predictions = pipeline.predict(test_data["complaint_text"])

    accuracy = float(
        accuracy_score(
            test_data[target_column],
            predictions,
        )
    )

    report = classification_report(
        test_data[target_column],
        predictions,
        output_dict=True,
        zero_division=0,
    )

    model_path = run_directory / f"{model_name}_candidate.joblib"
    metrics_path = run_directory / f"{model_name}_metrics.json"

    joblib.dump(pipeline, model_path)

    metrics = {
        "model_name": model_name,
        "target_column": target_column,
        "algorithm": "TF-IDF (1,2 grams) + LogisticRegression",
        "random_seed": RANDOM_SEED,
        "train_records": int(len(train_data)),
        "holdout_records": int(len(test_data)),
        "accuracy": accuracy,
        "classification_report": report,
        "artifact_file": str(model_path),
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return metrics


def main() -> None:
    arguments = parse_arguments()

    if not 0 < arguments.test_size < 1:
        fail("--test-size must be strictly between 0 and 1.")

    if arguments.min_records < 2:
        fail("--min-records must be at least 2.")

    if arguments.min_examples_per_class < 2:
        fail("--min-examples-per-class must be at least 2.")

    manifest = load_and_verify_manifest(
        csv_path=arguments.csv,
        manifest_path=arguments.manifest,
    )

    dataset = load_and_validate_dataset(arguments.csv)

    class_counts = validate_training_readiness(
        dataset=dataset,
        min_records=arguments.min_records,
        min_examples_per_class=arguments.min_examples_per_class,
    )

    primary_target = "final_category"

    train_data, test_data = train_test_split(
        dataset,
        test_size=arguments.test_size,
        random_state=RANDOM_SEED,
        stratify=dataset[primary_target],
    )

    run_timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_id = f"candidate-run-{run_timestamp}"

    run_directory = arguments.output_dir / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    model_metrics = []

    for model_name, target_column in MODEL_TARGETS.items():
        model_metrics.append(
            train_and_evaluate_model(
                model_name=model_name,
                target_column=target_column,
                train_data=train_data,
                test_data=test_data,
                run_directory=run_directory,
            )
        )

    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Candidate-only offline retraining. "
            "No active production model artifacts were modified."
        ),
        "source_dataset": {
            "csv_file": str(arguments.csv),
            "manifest_file": str(arguments.manifest),
            "dataset_version": manifest.get("dataset_version"),
            "dataset_sha256": manifest.get("sha256"),
            "manifest_schema_version": manifest.get("schema_version"),
            "manifest_record_count": manifest.get("record_count"),
        },
        "data_validation": {
            "valid_records_used": int(len(dataset)),
            "minimum_records_required": arguments.min_records,
            "minimum_examples_per_class": (
                arguments.min_examples_per_class
            ),
            "class_counts": class_counts,
        },
        "split": {
            "strategy": "Stratified holdout on final_category",
            "random_seed": RANDOM_SEED,
            "test_size": arguments.test_size,
            "train_records": int(len(train_data)),
            "holdout_records": int(len(test_data)),
        },
        "models": model_metrics,
        "approval_status": "Candidate only — not approved for deployment",
    }

    metadata_path = run_directory / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print("CANDIDATE TRAINING SUCCESS")
    print(f"Run directory: {run_directory}")
    print(f"Records used: {len(dataset)}")
    print(f"Train records: {len(train_data)}")
    print(f"Holdout records: {len(test_data)}")

    for metrics in model_metrics:
        print(
            f"{metrics['model_name']} accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

    print("Active production artifacts were not modified.")


if __name__ == "__main__":
    main()