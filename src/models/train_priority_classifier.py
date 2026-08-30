from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
TEST_PATH = Path("data/processed/test.csv")

ARTIFACT_DIR = Path("artifacts/priority_model")
MODEL_PATH = ARTIFACT_DIR / "priority_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_PATH = ARTIFACT_DIR / "classification_report.txt"
PREVIEW_PATH = ARTIFACT_DIR / "test_prediction_preview.csv"

TARGET_COLUMN = "priority"

NUMERIC_FEATURES = [
    "affected_population",
    "safety_flag",
    "repeat_count",
]

CATEGORICAL_FEATURES = [
    "category",
    "location_type",
    "language",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    df = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, "complaint_text"])
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    return df.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict:
    predictions = model.predict(features)

    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(labels, predictions, average="weighted")), 4),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_data(TRAIN_PATH)
    validation_df = load_data(VALIDATION_PATH)
    test_df = load_data(TEST_PATH)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    x_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = build_pipeline()
    model.fit(x_train, y_train)

    validation_metrics = calculate_metrics(model, x_validation, y_validation)

    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_model = build_pipeline()
    final_model.fit(
        combined_train_df[FEATURE_COLUMNS],
        combined_train_df[TARGET_COLUMN],
    )

    test_predictions = final_model.predict(x_test)
    test_probabilities = final_model.predict_proba(x_test)

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, test_predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, test_predictions, average="weighted")), 4),
    }

    priority_labels = list(final_model.named_steps["classifier"].classes_)
    report = classification_report(
        y_test,
        test_predictions,
        labels=priority_labels,
        digits=4,
        zero_division=0,
    )

    results = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_records": len(train_df),
        "validation_records": len(validation_df),
        "test_records": len(test_df),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "priority_labels": priority_labels,
    }

    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")

    preview = test_df[
        [
            "complaint_id",
            "complaint_text",
            "category",
            "affected_population",
            "safety_flag",
            "repeat_count",
        ]
    ].head(15).copy()

    preview["actual_priority"] = y_test.head(15).to_numpy()
    preview["predicted_priority"] = test_predictions[:15]
    preview["confidence"] = test_probabilities.max(axis=1)[:15].round(4)

    preview.to_csv(PREVIEW_PATH, index=False, encoding="utf-8")

    print("Priority classifier training completed.\n")
    print("Validation metrics:")
    print(f"Accuracy:    {validation_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {validation_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {validation_metrics['weighted_f1']:.4f}")

    print("\nFinal test metrics:")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved detailed report: {REPORT_PATH}")
    print(f"Saved prediction preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()