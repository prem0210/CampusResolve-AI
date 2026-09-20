from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from apps.api.core.dependencies import get_current_user
from apps.api.main import app
from src.database.database import Base, get_db
from src.database.models import (
    AuditLog,
    Complaint,
    ComplaintStatusHistory,
    User,
)


@asynccontextmanager
async def workflow_test_lifespan(app_instance: object):
    yield


def make_test_complaint(reference: str) -> Complaint:
    return Complaint(
        complaint_reference=reference,
        complaint_text="Water is leaking in Hostel Block A.",
        language="en",
        location_type="Hostel",
        specific_location="Hostel Block A",
        affected_population=20,
        safety_flag=False,
        repeat_count=0,
        predicted_category="Water and Plumbing",
        category_confidence=0.90,
        assigned_department="Plumbing and Civil Maintenance",
        predicted_priority="High",
        priority_confidence=0.85,
        estimated_resolution_hours=12.0,
        prediction_interval_plus_minus_hours=6.0,
        duplicate_threshold=0.70,
        possible_duplicate=False,
        status="Open",
    )


@pytest.fixture
def db_session(tmp_path: Path) -> Generator[Session, None, None]:
    database_path = tmp_path / "workflow_api_test.db"

    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(bind=engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    db = session_factory()

    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def staff_user(db_session: Session) -> User:
    user = User(
        full_name="Workflow Test Staff",
        email="workflow.staff@example.com",
        password_hash="test-password-hash",
        role="Staff",
        department_id=None,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    return user


@pytest.fixture
def client(
    db_session: Session,
    staff_user: User,
) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    def override_get_current_user() -> User:
        return staff_user

    original_lifespan = app.router.lifespan_context

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[
        get_current_user
    ] = override_get_current_user
    app.router.lifespan_context = workflow_test_lifespan

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    app.router.lifespan_context = original_lifespan


def test_status_route_rejects_open_to_resolved_transition(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0001")
    db_session.add(complaint)
    db_session.commit()

    response = client.patch(
        "/complaints/CR-TEST-0001",
        json={"status": "Resolved"},
    )

    assert response.status_code == 400
    assert "Invalid status transition" in response.json()["detail"]

    db_session.refresh(complaint)

    assert complaint.status == "Open"

def test_status_route_accepts_open_to_in_progress_and_records_history(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0002")
    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        "/complaints/CR-TEST-0002",
        json={"status": "In Progress"},
    )

    assert response.status_code == 200

    db_session.refresh(complaint)

    assert complaint.status == "In Progress"

    history = (
        db_session.query(ComplaintStatusHistory)
        .filter(
            ComplaintStatusHistory.complaint_id
            == complaint.id
        )
        .one()
    )

    assert history.old_status == "Open"
    assert history.new_status == "In Progress"

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == complaint.id,
        )
        .order_by(AuditLog.created_at.desc())
        .first()
    )

    assert audit_log is not None
    assert audit_log.actor_user_id == staff_user.id