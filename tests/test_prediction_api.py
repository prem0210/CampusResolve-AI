from contextlib import asynccontextmanager
from datetime import datetime

from fastapi.testclient import TestClient

from apps.api.main import app, prediction_service
from src.database.database import get_db
from src.database.models import Complaint


MOCK_PREDICTION = {
    "predicted_category": "Water and Plumbing",
    "category_confidence": 0.91,
    "assigned_department": "Plumbing and Civil Maintenance",
    "predicted_priority": "High",
    "priority_confidence": 0.88,
    "estimated_resolution_hours": 9.5,
    "prediction_interval_plus_minus_hours": 4.25,
    "duplicate_threshold": 0.70,
    "possible_duplicate": True,
    "duplicate_candidates": [
        {
            "complaint_id": "CMP-0001",
            "complaint_text": "Water is leaking in Hostel Block B.",
            "category": "Water and Plumbing",
            "priority": "High",
            "similarity_score": 0.86,
        }
    ],
    "category_explanation": {
        "predicted_category": "Water and Plumbing",
        "confidence": 0.91,
        "top_features": [
            {
                "feature": "word:water",
                "contribution": 0.45,
            }
        ],
    },
    "priority_explanation": {
        "predicted_priority": "High",
        "confidence": 0.88,
        "top_features": [
            {
                "feature": "numeric__safety_flag",
                "shap_value": 0.62,
                "direction": "increases priority likelihood",
            }
        ],
    },
    "explanation": (
        "The complaint was classified as Water and Plumbing because of "
        "water-related wording. Safety indicators contributed to High priority."
    ),
}

VALID_PAYLOAD = {
    "complaint_text": "Hostel Block B has a water leak and the floor is slippery.",
    "language": "en",
    "location_type": "Hostel",
    "specific_location": "Hostel Block B",
    "affected_population": 120,
    "safety_flag": 1,
    "repeat_count": 2,
}


class FakePredictionService:
    def is_ready(self) -> bool:
        return True

    def load_artifacts(self) -> None:
        pass

    def predict(self, payload: dict) -> dict:
        return MOCK_PREDICTION.copy()


class FakeDatabase:
    def __init__(self) -> None:
        self.complaints: list[Complaint] = []

    def add(self, complaint: Complaint) -> None:
        self.complaints.append(complaint)

    def commit(self) -> None:
        pass

    def refresh(self, complaint: Complaint) -> None:
        complaint.created_at = datetime.now()
        complaint.updated_at = datetime.now()

    def rollback(self) -> None:
        pass

    def scalar(self, statement):
        return None


def get_test_db():
    yield FakeDatabase()


@asynccontextmanager
async def mock_lifespan(app_instance):
    yield


def create_test_client() -> TestClient:
    app.dependency_overrides[get_db] = get_test_db
    app.router.lifespan_context = mock_lifespan

    original_service = prediction_service
    fake_service = FakePredictionService()

    prediction_service.is_ready = fake_service.is_ready
    prediction_service.load_artifacts = fake_service.load_artifacts
    prediction_service.predict = fake_service.predict

    return TestClient(app)


def cleanup_test_client() -> None:
    app.dependency_overrides.clear()


def test_predict_endpoint_with_mocked_model() -> None:
    client = create_test_client()

    try:
        response = client.post("/predict", json=VALID_PAYLOAD)

        assert response.status_code == 200

        body = response.json()

        assert body["predicted_category"] == "Water and Plumbing"
        assert body["assigned_department"] == "Plumbing and Civil Maintenance"
        assert body["predicted_priority"] == "High"
        assert body["possible_duplicate"] is True
        assert len(body["duplicate_candidates"]) == 1
        assert body["duplicate_candidates"][0]["complaint_id"] == "CMP-0001"
    finally:
        cleanup_test_client()


def test_create_complaint_endpoint_returns_created_response() -> None:
    client = create_test_client()

    try:
        response = client.post("/complaints", json=VALID_PAYLOAD)

        assert response.status_code == 201

        body = response.json()

        assert body["complaint_reference"].startswith("CR-")
        assert body["status"] == "Open"
        assert body["predicted_category"] == "Water and Plumbing"
        assert body["predicted_priority"] == "High"
        assert body["created_at"]
    finally:
        cleanup_test_client()


def test_predict_endpoint_rejects_invalid_payload() -> None:
    client = create_test_client()

    try:
        invalid_payload = VALID_PAYLOAD.copy()
        invalid_payload["complaint_text"] = "bad"

        response = client.post("/predict", json=invalid_payload)

        assert response.status_code == 422
    finally:
        cleanup_test_client()