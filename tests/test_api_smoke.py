import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from src.database.database import Base, engine


@pytest.fixture(autouse=True)
def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)


def test_root_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint_has_expected_shape() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert "models_loaded" in body
    assert "stored_complaints" in body