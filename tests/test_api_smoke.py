from fastapi.testclient import TestClient

from apps.api.main import app


def test_root_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "CampusResolve-AI API is running."
    assert response.json()["docs"] == "/docs"


def test_health_endpoint_has_expected_shape() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["service"] == "CampusResolve-AI"
    assert isinstance(body["models_loaded"], bool)
    assert isinstance(body["stored_complaints"], int)
    assert body["status"] in {"healthy", "starting"}