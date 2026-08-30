from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.explainability.explain_predictions import (
    explain_category_prediction,
    explain_priority_prediction,
    generate_human_readable_explanation,
)

CATEGORY_MODEL_PATH = Path("artifacts/category_model/category_classifier.joblib")
PRIORITY_MODEL_PATH = Path("artifacts/priority_model/priority_classifier.joblib")
RESOLUTION_MODEL_PATH = Path(
    "artifacts/resolution_model/resolution_time_regressor.joblib"
)
RESOLUTION_METRICS_PATH = Path("artifacts/resolution_model/metrics.json")

FAISS_INDEX_PATH = Path("artifacts/faiss_index/complaints.index")
FAISS_METADATA_PATH = Path("artifacts/faiss_index/complaint_metadata.csv")
FAISS_CONFIG_PATH = Path("artifacts/faiss_index/duplicate_detection_config.json")

TOP_K = 5

DEPARTMENT_MAPPING = {
    "Electrical and Power": "Electrical Maintenance",
    "Water and Plumbing": "Plumbing and Civil Maintenance",
    "Hostel and Accommodation": "Hostel Administration",
    "Wi-Fi and IT Services": "IT Support and Network Cell",
    "Classroom and Laboratory": "Academic Infrastructure",
    "Cleanliness and Waste": "Housekeeping and Sanitation",
    "Safety and Security": "Security Office",
    "Transport and Parking": "Transport Cell",
    "Administration and Documents": "Administrative Office",
    "Food and Canteen": "Canteen Committee",
}


class PredictionService:
    def __init__(self) -> None:
        self.category_model: Any | None = None
        self.priority_model: Any | None = None
        self.resolution_model: Any | None = None
        self.embedding_model: SentenceTransformer | None = None
        self.faiss_index: faiss.Index | None = None
        self.faiss_metadata: pd.DataFrame | None = None
        self.duplicate_threshold: float = 0.75
        self.resolution_interval: float = 24.0

    def load_artifacts(self) -> None:
        paths = [
            CATEGORY_MODEL_PATH,
            PRIORITY_MODEL_PATH,
            RESOLUTION_MODEL_PATH,
            RESOLUTION_METRICS_PATH,
            FAISS_INDEX_PATH,
            FAISS_METADATA_PATH,
            FAISS_CONFIG_PATH,
        ]

        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Required model artifacts are missing: " + ", ".join(missing)
            )

        self.category_model = joblib.load(CATEGORY_MODEL_PATH)
        self.priority_model = joblib.load(PRIORITY_MODEL_PATH)
        self.resolution_model = joblib.load(RESOLUTION_MODEL_PATH)

        resolution_metrics = json.loads(
            RESOLUTION_METRICS_PATH.read_text(encoding="utf-8")
        )
        self.resolution_interval = float(
            resolution_metrics["prediction_interval"]["plus_minus_hours"]
        )

        faiss_config = json.loads(FAISS_CONFIG_PATH.read_text(encoding="utf-8"))
        self.duplicate_threshold = float(
            faiss_config["metrics"]["selected_threshold"]
        )

        model_name = faiss_config["model_name"]
        self.embedding_model = SentenceTransformer(model_name)
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.faiss_metadata = pd.read_csv(FAISS_METADATA_PATH).fillna("")

    def is_ready(self) -> bool:
        return all(
            [
                self.category_model is not None,
                self.priority_model is not None,
                self.resolution_model is not None,
                self.embedding_model is not None,
                self.faiss_index is not None,
                self.faiss_metadata is not None,
            ]
        )

    def _search_duplicates(self, complaint_text: str) -> list[dict[str, Any]]:
        if not self.embedding_model or not self.faiss_index or self.faiss_metadata is None:
            raise RuntimeError("Duplicate-detection artifacts are not loaded.")

        embedding = self.embedding_model.encode(
            [complaint_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.faiss_index.search(embedding, TOP_K)
        candidates = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position == -1:
                continue

            row = self.faiss_metadata.iloc[int(index_position)]
            candidates.append(
                {
                    "complaint_id": str(row["complaint_id"]),
                    "complaint_text": str(row["complaint_text"]),
                    "category": str(row["category"]),
                    "priority": str(row["priority"]),
                    "similarity_score": round(float(score), 4),
                }
            )

        return candidates

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Prediction service is not ready.")

        complaint_text = str(payload["complaint_text"]).strip()

        category = str(self.category_model.predict([complaint_text])[0])
        category_probability = float(
            self.category_model.predict_proba([complaint_text])[0].max()
        )
        department = DEPARTMENT_MAPPING[category]

        priority_features = {
            "affected_population": payload["affected_population"],
            "safety_flag": payload["safety_flag"],
            "repeat_count": payload["repeat_count"],
            "category": category,
            "location_type": payload["location_type"],
            "language": payload["language"],
        }

        priority = str(self.priority_model.predict(
            pd.DataFrame([priority_features])
        )[0])
        priority_probability = float(
            self.priority_model.predict_proba(
                pd.DataFrame([priority_features])
            )[0].max()
        )

        resolution_features = {
            **priority_features,
            "priority": priority,
        }
        estimated_resolution_hours = float(
            self.resolution_model.predict(
                pd.DataFrame([resolution_features])
            )[0]
        )

        category_explanation = explain_category_prediction(
            self.category_model,
            complaint_text,
        )
        priority_explanation = explain_priority_prediction(
            self.priority_model,
            priority_features,
        )
        explanation = generate_human_readable_explanation(
            category_explanation,
            priority_explanation,
        )

        duplicate_candidates = self._search_duplicates(complaint_text)
        possible_duplicate = bool(
            duplicate_candidates
            and duplicate_candidates[0]["similarity_score"]
            >= self.duplicate_threshold
        )

        return {
            "predicted_category": category,
            "category_confidence": round(category_probability, 4),
            "assigned_department": department,
            "predicted_priority": priority,
            "priority_confidence": round(priority_probability, 4),
            "estimated_resolution_hours": round(estimated_resolution_hours, 2),
            "prediction_interval_plus_minus_hours": round(
                self.resolution_interval,
                2,
            ),
            "duplicate_threshold": round(self.duplicate_threshold, 2),
            "possible_duplicate": possible_duplicate,
            "duplicate_candidates": duplicate_candidates,
            "category_explanation": category_explanation,
            "priority_explanation": priority_explanation,
            "explanation": explanation,
        }