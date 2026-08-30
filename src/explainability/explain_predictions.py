from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


CATEGORY_MODEL_PATH = Path(
    "artifacts/category_model/category_classifier.joblib"
)
PRIORITY_MODEL_PATH = Path(
    "artifacts/priority_model/priority_classifier.joblib"
)

TOP_FEATURES = 5


def load_models() -> tuple[Any, Any]:
    if not CATEGORY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Category model not found: {CATEGORY_MODEL_PATH}"
        )

    if not PRIORITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Priority model not found: {PRIORITY_MODEL_PATH}"
        )

    category_model = joblib.load(CATEGORY_MODEL_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)

    return category_model, priority_model


def get_category_feature_names(category_model: Any) -> np.ndarray:
    feature_union = category_model.named_steps["features"]
    word_vectorizer = feature_union.transformer_list[0][1]
    char_vectorizer = feature_union.transformer_list[1][1]

    word_features = [
        f"word:{feature}"
        for feature in word_vectorizer.get_feature_names_out()
    ]
    char_features = [
        f"char:{feature}"
        for feature in char_vectorizer.get_feature_names_out()
    ]

    return np.array(word_features + char_features)


def explain_category_prediction(
    category_model: Any,
    complaint_text: str,
    top_n: int = TOP_FEATURES,
) -> dict[str, Any]:
    features = category_model.named_steps["features"]
    classifier = category_model.named_steps["classifier"]

    transformed = features.transform([complaint_text])
    predicted_category = str(category_model.predict([complaint_text])[0])
    probabilities = category_model.predict_proba([complaint_text])[0]

    class_index = list(classifier.classes_).index(predicted_category)
    class_coefficients = classifier.coef_[class_index]
    contributions = transformed.multiply(class_coefficients).toarray().ravel()

    feature_names = get_category_feature_names(category_model)
    positive_indices = np.where(contributions > 0)[0]
    ranked_indices = positive_indices[
        np.argsort(contributions[positive_indices])[::-1]
    ][:top_n]

    influential_features = [
        {
            "feature": str(feature_names[index]),
            "contribution": round(float(contributions[index]), 4),
        }
        for index in ranked_indices
    ]

    confidence = round(float(np.max(probabilities)), 4)

    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "top_features": influential_features,
    }


def format_priority_features(
    feature_names: np.ndarray,
    shap_values: np.ndarray,
    top_n: int,
) -> list[dict[str, Any]]:
    ranked_indices = np.argsort(np.abs(shap_values))[::-1][:top_n]

    return [
        {
            "feature": str(feature_names[index]),
            "shap_value": round(float(shap_values[index]), 4),
            "direction": "increases priority likelihood"
            if shap_values[index] > 0
            else "decreases priority likelihood",
        }
        for index in ranked_indices
    ]


def explain_priority_prediction(
    priority_model: Any,
    complaint_features: dict[str, Any],
    top_n: int = TOP_FEATURES,
) -> dict[str, Any]:
    input_df = pd.DataFrame([complaint_features])

    preprocessor = priority_model.named_steps["preprocessor"]
    classifier = priority_model.named_steps["classifier"]

    transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    predicted_priority = str(priority_model.predict(input_df)[0])
    probabilities = priority_model.predict_proba(input_df)[0]

    class_index = list(classifier.classes_).index(predicted_priority)

    feature_values = transformed.toarray().ravel()
    class_coefficients = np.asarray(
        classifier.coef_[class_index]
    ).ravel()

    contributions = feature_values * class_coefficients

    top_features = format_priority_features(
        feature_names=feature_names,
        shap_values=contributions,
        top_n=top_n,
    )

    confidence = round(float(np.max(probabilities)), 4)

    return {
        "predicted_priority": predicted_priority,
        "confidence": confidence,
        "top_features": top_features,
    }


def generate_human_readable_explanation(
    category_explanation: dict[str, Any],
    priority_explanation: dict[str, Any],
) -> str:
    category_terms = [
        item["feature"].replace("word:", "")
        for item in category_explanation["top_features"]
        if item["feature"].startswith("word:")
    ]

    category_terms = category_terms[:3]
    category_hint = (
        ", ".join(category_terms)
        if category_terms
        else "the complaint wording"
    )

    priority_factors = [
        item["feature"]
        .replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_", " ")
        for item in priority_explanation["top_features"][:3]
    ]

    priority_hint = ", ".join(priority_factors)

    return (
        f"The complaint was classified as "
        f"'{category_explanation['predicted_category']}' "
        f"because of signals including {category_hint}. "
        f"The predicted priority is "
        f"'{priority_explanation['predicted_priority']}', "
        f"primarily influenced by {priority_hint}."
    )


def explain_complaint(
    complaint_text: str,
    priority_features: dict[str, Any],
) -> dict[str, Any]:
    category_model, priority_model = load_models()

    category_explanation = explain_category_prediction(
        category_model=category_model,
        complaint_text=complaint_text,
    )

    priority_explanation = explain_priority_prediction(
        priority_model=priority_model,
        complaint_features=priority_features,
    )

    narrative = generate_human_readable_explanation(
        category_explanation=category_explanation,
        priority_explanation=priority_explanation,
    )

    return {
        "category_explanation": category_explanation,
        "priority_explanation": priority_explanation,
        "narrative": narrative,
    }