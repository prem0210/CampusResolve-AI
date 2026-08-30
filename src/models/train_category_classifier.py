from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
TEST_PATH = Path("data/processed/test.csv")

ARTIFACT_DIR = Path("artifacts/category_model")
MODEL_PATH = ARTIFACT_DIR / "category_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_PATH = ARTIFACT_DIR / "classification_report.txt"

TEXT_COLUMN = "normalized_text"
TARGET_COLUMN = "category"

MODEL_CONFIGS = {
    "word_tfidf_logreg": {
        "features": FeatureUnion(
            transformer_list=[
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
            ]
        ),
        "classifier": LogisticRegression(
            C=2.0,
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    },
    "word_only_logreg": {
        "features": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        ),
        "classifier": LogisticRegression(
            C=2.0,
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    },
}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    df = pd.read_csv(path)
    required_columns = {TEXT_COLUMN, TARGET_COLUMN}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    return df[df[TEXT_COLUMN].ne("")].reset_index(drop=True)


def build_pipeline(config: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", config["features"]),
            ("classifier", config["classifier"]),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_data: pd.Series,
    y_true: pd.Series,
) -> dict:
    predictions = model.predict(x_data)

    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, predictions, average="weighted")), 4),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_data(TRAIN_PATH)
    validation_df = load_data(VALIDATION_PATH)
    test_df = load_data(TEST_PATH)

    x_train = train_df[TEXT_COLUMN]
    y_train = train_df[TARGET_COLUMN]

    x_validation = validation_df[TEXT_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]

    x_test = test_df[TEXT_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    validation_results = {}
    trained_models = {}

    print("Training candidate category-classification models...\n")

    for model_name, config in MODEL_CONFIGS.items():
        pipeline = build_pipeline(config)
        pipeline.fit(x_train, y_train)

        metrics = evaluate_model(pipeline, x_validation, y_validation)
        validation_results[model_name] = metrics
        trained_models[model_name] = pipeline

        print(f"{model_name}")
        print(f"  Validation Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Validation Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"  Validation Weighted F1: {metrics['weighted_f1']:.4f}\n")

    best_model_name = max(
        validation_results,
        key=lambda name: validation_results[name]["macro_f1"],
    )
    best_config = MODEL_CONFIGS[best_model_name]

    print(f"Selected model: {best_model_name}")

    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    x_train_final = combined_train_df[TEXT_COLUMN]
    y_train_final = combined_train_df[TARGET_COLUMN]

    final_model = build_pipeline(best_config)
    final_model.fit(x_train_final, y_train_final)

    test_predictions = final_model.predict(x_test)
    test_probabilities = final_model.predict_proba(x_test)

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, test_predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, test_predictions, average="weighted")), 4),
    }

    category_labels = list(final_model.named_steps["classifier"].classes_)
    report = classification_report(
        y_test,
        test_predictions,
        labels=category_labels,
        digits=4,
        zero_division=0,
    )

    results = {
        "selected_model": best_model_name,
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "train_records": len(train_df),
        "validation_records": len(validation_df),
        "test_records": len(test_df),
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "categories": category_labels,
    }

    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")

    prediction_preview = pd.DataFrame(
        {
            "complaint_text": test_df["complaint_text"].head(10),
            "actual_category": y_test.head(10),
            "predicted_category": test_predictions[:10],
            "confidence": test_probabilities.max(axis=1)[:10].round(4),
        }
    )
    prediction_preview.to_csv(
        ARTIFACT_DIR / "test_prediction_preview.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nFinal test metrics:")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved detailed report: {REPORT_PATH}")


if __name__ == "__main__":
    main()