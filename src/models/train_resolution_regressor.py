from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
TEST_PATH = Path("data/processed/test.csv")

ARTIFACT_DIR = Path("artifacts/resolution_model")
MODEL_PATH = ARTIFACT_DIR / "resolution_time_regressor.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREVIEW_PATH = ARTIFACT_DIR / "test_prediction_preview.csv"

TARGET_COLUMN = "resolution_time_hours"

NUMERIC_FEATURES = [
    "affected_population",
    "safety_flag",
    "repeat_count",
]

CATEGORICAL_FEATURES = [
    "category",
    "location_type",
    "language",
    "priority",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    df = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, "complaint_id", "complaint_text"])
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df = df[df[TARGET_COLUMN] > 0].reset_index(drop=True)

    return df


def build_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    regressor = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = r2_score(actual, predicted)

    return {
        "mae_hours": round(float(mae), 4),
        "rmse_hours": round(rmse, 4),
        "r2_score": round(float(r2), 4),
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

    baseline_prediction = np.full(
        shape=len(y_validation),
        fill_value=float(y_train.median()),
    )
    baseline_validation_metrics = calculate_metrics(y_validation, baseline_prediction)

    model = build_model()
    model.fit(x_train, y_train)
    validation_predictions = model.predict(x_validation)
    validation_metrics = calculate_metrics(y_validation, validation_predictions)

    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)

    final_model = build_model()
    final_model.fit(
        combined_train_df[FEATURE_COLUMNS],
        combined_train_df[TARGET_COLUMN],
    )

    test_predictions = final_model.predict(x_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    prediction_error = np.abs(y_test.to_numpy() - test_predictions)
    interval_width = float(np.quantile(prediction_error, 0.90))

    preview = test_df[
        [
            "complaint_id",
            "complaint_text",
            "category",
            "priority",
            "affected_population",
            "safety_flag",
            "repeat_count",
            TARGET_COLUMN,
        ]
    ].head(20).copy()

    preview["predicted_resolution_time_hours"] = test_predictions[:20].round(2)
    preview["absolute_error_hours"] = (
        preview[TARGET_COLUMN].to_numpy()
        - preview["predicted_resolution_time_hours"].to_numpy()
    ).astype(float)
    preview["absolute_error_hours"] = preview["absolute_error_hours"].abs().round(2)

    preview.to_csv(PREVIEW_PATH, index=False, encoding="utf-8")

    results = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_records": len(train_df),
        "validation_records": len(validation_df),
        "test_records": len(test_df),
        "baseline_validation_metrics": baseline_validation_metrics,
        "model_validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "prediction_interval": {
            "coverage_target": 0.90,
            "plus_minus_hours": round(interval_width, 2),
            "method": "90th percentile absolute test residual",
        },
    }

    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Resolution-time regression completed.\n")

    print("Validation baseline metrics:")
    print(f"MAE (hours):  {baseline_validation_metrics['mae_hours']:.4f}")
    print(f"RMSE (hours): {baseline_validation_metrics['rmse_hours']:.4f}")
    print(f"R² score:     {baseline_validation_metrics['r2_score']:.4f}")

    print("\nValidation model metrics:")
    print(f"MAE (hours):  {validation_metrics['mae_hours']:.4f}")
    print(f"RMSE (hours): {validation_metrics['rmse_hours']:.4f}")
    print(f"R² score:     {validation_metrics['r2_score']:.4f}")

    print("\nFinal test metrics:")
    print(f"MAE (hours):  {test_metrics['mae_hours']:.4f}")
    print(f"RMSE (hours): {test_metrics['rmse_hours']:.4f}")
    print(f"R² score:     {test_metrics['r2_score']:.4f}")
    print(f"90% prediction interval: ±{interval_width:.2f} hours")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved prediction preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()