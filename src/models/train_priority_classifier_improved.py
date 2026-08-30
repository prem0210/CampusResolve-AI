from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.priority_risk_features import (
    extract_text_risk_features,
    get_risk_feature_names,
)

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
TEST_PATH = Path("data/processed/test.csv")

ARTIFACT_DIR = Path("artifacts/priority_model_improved")
MODEL_PATH = ARTIFACT_DIR / "priority_classifier_improved.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_PATH = ARTIFACT_DIR / "classification_report.txt"
PREVIEW_PATH = ARTIFACT_DIR / "test_prediction_preview.csv"

TARGET_COLUMN = "priority"
TEXT_COLUMN = "complaint_text"

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

STRUCTURED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class PriorityFeaturePipeline:
    def __init__(self) -> None:
        self.structured_pipeline = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    NUMERIC_FEATURES,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ],
            remainder="drop",
        )

        self.text_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )

        self.risk_scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame):
        structured_matrix = self.structured_pipeline.fit_transform(
            df[STRUCTURED_FEATURES]
        )

        text_matrix = self.text_vectorizer.fit_transform(
            df[TEXT_COLUMN].fillna("").astype(str)
        )

        risk_df = extract_text_risk_features(df[TEXT_COLUMN])
        risk_matrix = csr_matrix(
            self.risk_scaler.fit_transform(risk_df)
        )

        return hstack(
            [structured_matrix, text_matrix, risk_matrix],
            format="csr",
        )

    def transform(self, df: pd.DataFrame):
        structured_matrix = self.structured_pipeline.transform(
            df[STRUCTURED_FEATURES]
        )

        text_matrix = self.text_vectorizer.transform(
            df[TEXT_COLUMN].fillna("").astype(str)
        )

        risk_df = extract_text_risk_features(df[TEXT_COLUMN])
        risk_matrix = csr_matrix(self.risk_scaler.transform(risk_df))

        return hstack(
            [structured_matrix, text_matrix, risk_matrix],
            format="csr",
        )


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    df = pd.read_csv(path)

    required_columns = set(
        STRUCTURED_FEATURES + [TEXT_COLUMN, TARGET_COLUMN, "complaint_id"]
    )
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()

    return df.reset_index(drop=True)


def calculate_metrics(
    actual: pd.Series,
    predicted,
) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro")),
            4,
        ),
        "weighted_f1": round(
            float(f1_score(actual, predicted, average="weighted")),
            4,
        ),
    }


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_data(TRAIN_PATH)
    validation_df = load_data(VALIDATION_PATH)
    test_df = load_data(TEST_PATH)

    feature_pipeline = PriorityFeaturePipeline()
    x_train = feature_pipeline.fit_transform(train_df)
    x_validation = feature_pipeline.transform(validation_df)

    classifier = build_classifier()
    classifier.fit(x_train, train_df[TARGET_COLUMN])

    validation_predictions = classifier.predict(x_validation)
    validation_metrics = calculate_metrics(
        validation_df[TARGET_COLUMN],
        validation_predictions,
    )

    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)

    final_feature_pipeline = PriorityFeaturePipeline()
    x_train_final = final_feature_pipeline.fit_transform(combined_train_df)
    x_test = final_feature_pipeline.transform(test_df)

    final_classifier = build_classifier()
    final_classifier.fit(
        x_train_final,
        combined_train_df[TARGET_COLUMN],
    )

    test_predictions = final_classifier.predict(x_test)
    test_probabilities = final_classifier.predict_proba(x_test)

    test_metrics = calculate_metrics(
        test_df[TARGET_COLUMN],
        test_predictions,
    )

    priority_labels = list(final_classifier.classes_)

    report = classification_report(
        test_df[TARGET_COLUMN],
        test_predictions,
        labels=priority_labels,
        digits=4,
        zero_division=0,
    )

    artifact = {
        "feature_pipeline": final_feature_pipeline,
        "classifier": final_classifier,
        "structured_features": STRUCTURED_FEATURES,
        "text_column": TEXT_COLUMN,
        "risk_feature_names": get_risk_feature_names(),
    }

    results = {
        "model_name": "structured_tfidf_risk_logistic_regression",
        "target_column": TARGET_COLUMN,
        "structured_features": STRUCTURED_FEATURES,
        "risk_feature_names": get_risk_feature_names(),
        "train_records": len(train_df),
        "validation_records": len(validation_df),
        "test_records": len(test_df),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "priority_labels": priority_labels,
        "baseline_test_metrics": {
            "accuracy": 0.3527,
            "macro_f1": 0.3469,
            "weighted_f1": 0.3489,
        },
    }

    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    preview = test_df[
        [
            "complaint_id",
            "complaint_text",
            "category",
            "safety_flag",
            "affected_population",
            "priority",
        ]
    ].head(20).copy()

    preview["predicted_priority"] = test_predictions[:20]
    preview["confidence"] = test_probabilities.max(axis=1)[:20].round(4)

    preview.to_csv(PREVIEW_PATH, index=False, encoding="utf-8")

    print("Improved priority classifier training completed.\n")

    print("Validation metrics:")
    print(f"Accuracy:    {validation_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {validation_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {validation_metrics['weighted_f1']:.4f}")

    print("\nFinal test metrics:")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    print("\nBaseline test macro F1: 0.3469")
    print(
        "Improved test macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()