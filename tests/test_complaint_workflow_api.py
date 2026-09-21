from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    ComplaintAssignment,
    ComplaintAssignmentHistory,
    ComplaintEscalation,
    ComplaintOwnership,
    ComplaintStatusHistory,
    ComplaintVerification,
    Department,
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
    department = Department(
        code="PLUMB",
        name="Plumbing and Civil Maintenance",
        description="Handles water and civil maintenance requests.",
        is_active=True,
    )

    db_session.add(department)
    db_session.commit()
    db_session.refresh(department)

    user = User(
        full_name="Workflow Test Staff",
        email="workflow.staff@example.com",
        password_hash="test-password-hash",
        role="Staff",
        department_id=department.id,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    return user

@pytest.fixture
def same_department_staff(
    db_session: Session,
    staff_user: User,
) -> User:
    user = User(
        full_name="Plumbing Colleague",
        email="plumbing.colleague@example.com",
        password_hash="test-password-hash",
        role="Staff",
        department_id=staff_user.department_id,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    return user

@pytest.fixture
def other_department_staff(
    db_session: Session,
    other_department: Department,
) -> User:
    user = User(
        full_name="Electrical Test Staff",
        email="electrical.staff@example.com",
        password_hash="test-password-hash",
        role="Staff",
        department_id=other_department.id,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    return user

@pytest.fixture
def other_department(db_session: Session) -> Department:
    department = Department(
        code="ELEC",
        name="Electrical Maintenance",
        description="Handles electrical maintenance requests.",
        is_active=True,
    )

    db_session.add(department)
    db_session.commit()
    db_session.refresh(department)

    return department

@pytest.fixture
def admin_user(db_session: Session) -> User:
    user = User(
        full_name="Workflow Test Admin",
        email="workflow.admin@example.com",
        password_hash="test-password-hash",
        role="Admin",
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

@pytest.fixture
def admin_client(
    db_session: Session,
    admin_user: User,
) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    def override_get_current_user() -> User:
        return admin_user

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

@pytest.fixture
def student_user(db_session: Session) -> User:
    user = User(
        full_name="Student A",
        email="student.a@example.com",
        password_hash="test-password-hash",
        role="Student",
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
def other_student_user(db_session: Session) -> User:
    user = User(
        full_name="Student B",
        email="student.b@example.com",
        password_hash="test-password-hash",
        role="Student",
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
def student_client(
    db_session: Session,
    student_user: User,
) -> Generator[TestClient, None, None]:
    yield from build_authenticated_client(
        db_session=db_session,
        authenticated_user=student_user,
    )


@pytest.fixture
def other_student_client(
    db_session: Session,
    other_student_user: User,
) -> Generator[TestClient, None, None]:
    yield from build_authenticated_client(
        db_session=db_session,
        authenticated_user=other_student_user,
    )

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
            AuditLog.entity_id == str(complaint.id),
        )
        .order_by(AuditLog.created_at.desc())
        .first()
    )

    assert audit_log is not None
    assert audit_log.actor_user_id == staff_user.id

def test_staff_cannot_update_complaint_assigned_to_another_department(
    client: TestClient,
    db_session: Session,
    other_department: Department,
    other_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0003")
    complaint.assigned_department = other_department.name

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_department_id=other_department.id,
        assigned_to_user_id=other_department_staff.id,
        assigned_by_user_id=other_department_staff.id,
        assignment_note="Assigned to electrical maintenance for testing.",
    )

    db_session.add(assignment)
    db_session.commit()

    response = client.patch(
        "/complaints/CR-TEST-0003",
        json={"status": "In Progress"},
    )

    assert response.status_code == 403

    db_session.refresh(complaint)

    assert complaint.status == "Open"

    history_count = (
        db_session.query(ComplaintStatusHistory)
        .filter(ComplaintStatusHistory.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == complaint.id,
        )
        .count()
    )

    assert history_count == 0
    assert audit_count == 0

def test_staff_can_update_complaint_assigned_to_them(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0004")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_to_user_id=staff_user.id,
        assigned_department_id=staff_user.department_id,
        assigned_by_user_id=staff_user.id,
        assignment_note="Assigned to the authenticated staff user for testing.",
    )

    db_session.add(assignment)
    db_session.commit()

    response = client.patch(
        "/complaints/CR-TEST-0004",
        json={"status": "In Progress"},
    )

    assert response.status_code == 200

    db_session.refresh(complaint)

    assert complaint.status == "In Progress"

    history = (
        db_session.query(ComplaintStatusHistory)
        .filter(ComplaintStatusHistory.complaint_id == complaint.id)
        .one()
    )

    assert history.old_status == "Open"
    assert history.new_status == "In Progress"

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "UPDATE_COMPLAINT_STATUS",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_staff_can_update_complaint_assigned_to_colleague_in_same_department(
    client: TestClient,
    db_session: Session,
    staff_user: User,
    same_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0005")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_to_user_id=same_department_staff.id,
        assigned_department_id=staff_user.department_id,
        assigned_by_user_id=staff_user.id,
        assignment_note="Assigned to a colleague in the same department.",
    )

    db_session.add(assignment)
    db_session.commit()

    response = client.patch(
        "/complaints/CR-TEST-0005",
        json={"status": "In Progress"},
    )

    assert response.status_code == 200

    db_session.refresh(complaint)

    assert complaint.status == "In Progress"

    history = (
        db_session.query(ComplaintStatusHistory)
        .filter(ComplaintStatusHistory.complaint_id == complaint.id)
        .one()
    )

    assert history.old_status == "Open"
    assert history.new_status == "In Progress"
    assert history.changed_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "UPDATE_COMPLAINT_STATUS",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_admin_can_update_complaint_assigned_to_another_department(
    admin_client: TestClient,
    db_session: Session,
    admin_user: User,
    other_department: Department,
    other_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0006")
    complaint.assigned_department = other_department.name

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_to_user_id=other_department_staff.id,
        assigned_department_id=other_department.id,
        assigned_by_user_id=admin_user.id,
        assignment_note="Assigned to electrical maintenance for admin override testing.",
    )

    db_session.add(assignment)
    db_session.commit()

    response = admin_client.patch(
        "/complaints/CR-TEST-0006",
        json={"status": "In Progress"},
    )

    assert response.status_code == 200

    db_session.refresh(complaint)

    assert complaint.status == "In Progress"

    history = (
        db_session.query(ComplaintStatusHistory)
        .filter(ComplaintStatusHistory.complaint_id == complaint.id)
        .one()
    )

    assert history.old_status == "Open"
    assert history.new_status == "In Progress"
    assert history.changed_by_user_id == admin_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "UPDATE_COMPLAINT_STATUS",
        )
        .one()
    )

    assert audit_log.actor_user_id == admin_user.id

def test_admin_can_reassign_complaint_across_departments(
    admin_client: TestClient,
    db_session: Session,
    admin_user: User,
    staff_user: User,
    other_department: Department,
    other_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0014")
    complaint.assigned_department = other_department.name

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    original_assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_to_user_id=other_department_staff.id,
        assigned_department_id=other_department.id,
        assigned_by_user_id=admin_user.id,
        assignment_note="Initial electrical assignment.",
    )

    db_session.add(original_assignment)
    db_session.commit()

    response = admin_client.put(
        f"/complaints/{complaint.complaint_reference}/assignment",
        json={
            "assigned_to_user_id": staff_user.id,
            "assigned_department_id": staff_user.department_id,
            "assignment_note": (
                "Reassigned by Admin to plumbing maintenance."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["assigned_to_user_id"] == staff_user.id
    assert payload["assigned_department_id"] == staff_user.department_id
    assert payload["assigned_by_user_id"] == admin_user.id
    assert payload["assignment_note"] == (
        "Reassigned by Admin to plumbing maintenance."
    )

    assignments = (
        db_session.query(ComplaintAssignment)
        .filter(ComplaintAssignment.complaint_id == complaint.id)
        .all()
    )

    assert len(assignments) == 1

    assignment = assignments[0]

    assert assignment.assigned_to_user_id == staff_user.id
    assert assignment.assigned_department_id == staff_user.department_id
    assert assignment.assigned_by_user_id == admin_user.id

    assignment_history = (
        db_session.query(ComplaintAssignmentHistory)
        .filter(
            ComplaintAssignmentHistory.complaint_id == complaint.id
        )
        .one()
    )

    assert assignment_history.previous_assigned_to_user_id == (
        other_department_staff.id
    )
    assert assignment_history.new_assigned_to_user_id == staff_user.id
    assert assignment_history.previous_department_id == other_department.id
    assert assignment_history.new_department_id == staff_user.department_id
    assert assignment_history.changed_by_user_id == admin_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "REASSIGN_COMPLAINT",
        )
        .one()
    )

    assert audit_log.actor_user_id == admin_user.id

def test_staff_can_self_assign_unassigned_complaint(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0011")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/assignment",
        json={
            "assigned_to_user_id": staff_user.id,
            "assigned_department_id": staff_user.department_id,
            "assignment_note": (
                "Staff member claimed this complaint from the triage queue."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["complaint_reference"] == complaint.complaint_reference
    assert payload["assigned_to_user_id"] == staff_user.id
    assert payload["assigned_department_id"] == staff_user.department_id
    assert payload["assigned_by_user_id"] == staff_user.id

    assignment = (
        db_session.query(ComplaintAssignment)
        .filter(ComplaintAssignment.complaint_id == complaint.id)
        .one()
    )

    assert assignment.assigned_to_user_id == staff_user.id
    assert assignment.assigned_department_id == staff_user.department_id
    assert assignment.assigned_by_user_id == staff_user.id

    assignment_history = (
        db_session.query(ComplaintAssignmentHistory)
        .filter(
            ComplaintAssignmentHistory.complaint_id == complaint.id
        )
        .one()
    )

    assert assignment_history.previous_assigned_to_user_id is None
    assert assignment_history.new_assigned_to_user_id == staff_user.id
    assert assignment_history.previous_department_id is None
    assert assignment_history.new_department_id == staff_user.department_id
    assert assignment_history.changed_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "ASSIGN_COMPLAINT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_staff_cannot_assign_complaint_to_colleague(
    client: TestClient,
    db_session: Session,
    staff_user: User,
    same_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0012")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/assignment",
        json={
            "assigned_to_user_id": same_department_staff.id,
            "assigned_department_id": staff_user.department_id,
            "assignment_note": (
                "Attempted colleague assignment; should be rejected."
            ),
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Staff users can assign a complaint only to their own account."
    )

    assignment_count = (
        db_session.query(ComplaintAssignment)
        .filter(ComplaintAssignment.complaint_id == complaint.id)
        .count()
    )

    history_count = (
        db_session.query(ComplaintAssignmentHistory)
        .filter(
            ComplaintAssignmentHistory.complaint_id == complaint.id
        )
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert assignment_count == 0
    assert history_count == 0
    assert audit_count == 0

def test_staff_cannot_claim_complaint_assigned_to_another_department(
    client: TestClient,
    db_session: Session,
    staff_user: User,
    other_department: Department,
    other_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0013")
    complaint.assigned_department = other_department.name

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    original_assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_to_user_id=other_department_staff.id,
        assigned_department_id=other_department.id,
        assigned_by_user_id=other_department_staff.id,
        assignment_note="Originally assigned to electrical maintenance.",
    )

    db_session.add(original_assignment)
    db_session.commit()

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/assignment",
        json={
            "assigned_to_user_id": staff_user.id,
            "assigned_department_id": staff_user.department_id,
            "assignment_note": (
                "Attempted cross-department claim; should be rejected."
            ),
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "You do not have access to this complaint."
    )

    assignment = (
        db_session.query(ComplaintAssignment)
        .filter(ComplaintAssignment.complaint_id == complaint.id)
        .one()
    )

    assert assignment.assigned_to_user_id == other_department_staff.id
    assert assignment.assigned_department_id == other_department.id
    assert assignment.assignment_note == (
        "Originally assigned to electrical maintenance."
    )

    history_count = (
        db_session.query(ComplaintAssignmentHistory)
        .filter(
            ComplaintAssignmentHistory.complaint_id == complaint.id
        )
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert history_count == 0
    assert audit_count == 0

@asynccontextmanager
async def workflow_test_lifespan(app_instance: object):
    yield


def build_authenticated_client(
    db_session: Session,
    authenticated_user: User,
) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    def override_get_current_user() -> User:
        return authenticated_user

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

def create_owned_complaint(
    db_session: Session,
    complaint_reference: str,
    owner: User,
) -> Complaint:
    complaint = make_test_complaint(complaint_reference)

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    ownership = ComplaintOwnership(
        complaint_id=complaint.id,
        submitted_by_user_id=owner.id,
    )

    db_session.add(ownership)
    db_session.commit()

    return complaint

def test_student_cannot_read_another_students_complaint(
    other_student_client: TestClient,
    db_session: Session,
    student_user: User,
) -> None:
    complaint = create_owned_complaint(
        db_session=db_session,
        complaint_reference="CR-TEST-0007",
        owner=student_user,
    )

    response = other_student_client.get(
        f"/complaints/{complaint.complaint_reference}"
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "You do not have access to this complaint."
    )

def test_student_cannot_update_complaint_status(
    student_client: TestClient,
    db_session: Session,
    student_user: User,
) -> None:
    complaint = create_owned_complaint(
        db_session=db_session,
        complaint_reference="CR-TEST-0008",
        owner=student_user,
    )

    response = student_client.patch(
        f"/complaints/{complaint.complaint_reference}",
        json={"status": "In Progress"},
    )

    assert response.status_code == 403

    db_session.refresh(complaint)

    assert complaint.status == "Open"

    history_count = (
        db_session.query(ComplaintStatusHistory)
        .filter(ComplaintStatusHistory.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert history_count == 0
    assert audit_count == 0

def test_student_can_read_their_own_complaint(
    student_client: TestClient,
    db_session: Session,
    student_user: User,
) -> None:
    complaint = create_owned_complaint(
        db_session=db_session,
        complaint_reference="CR-TEST-0009",
        owner=student_user,
    )

    response = student_client.get(
        f"/complaints/{complaint.complaint_reference}"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["complaint_reference"] == complaint.complaint_reference
    assert payload["status"] == "Open"

def test_admin_can_assign_unassigned_complaint_to_staff(
    admin_client: TestClient,
    db_session: Session,
    admin_user: User,
    other_department: Department,
    other_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0010")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = admin_client.put(
        f"/complaints/{complaint.complaint_reference}/assignment",
        json={
            "assigned_to_user_id": other_department_staff.id,
            "assigned_department_id": other_department.id,
            "assignment_note": (
                "Assigned by Admin for assignment API integration testing."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["complaint_reference"] == complaint.complaint_reference
    assert payload["assigned_to_user_id"] == other_department_staff.id
    assert payload["assigned_department_id"] == other_department.id
    assert payload["assigned_by_user_id"] == admin_user.id
    assert payload["assignment_note"] == (
        "Assigned by Admin for assignment API integration testing."
    )

    assignment = (
        db_session.query(ComplaintAssignment)
        .filter(ComplaintAssignment.complaint_id == complaint.id)
        .one()
    )

    assert assignment.assigned_to_user_id == other_department_staff.id
    assert assignment.assigned_department_id == other_department.id
    assert assignment.assigned_by_user_id == admin_user.id

    assignment_history = (
        db_session.query(ComplaintAssignmentHistory)
        .filter(
            ComplaintAssignmentHistory.complaint_id == complaint.id
        )
        .one()
    )

    assert assignment_history.previous_assigned_to_user_id is None
    assert assignment_history.new_assigned_to_user_id == (
        other_department_staff.id
    )
    assert assignment_history.previous_department_id is None
    assert assignment_history.new_department_id == other_department.id
    assert assignment_history.changed_by_user_id == admin_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "ASSIGN_COMPLAINT",
        )
        .one()
    )

    assert audit_log.actor_user_id == admin_user.id

def test_staff_can_create_escalation_for_accessible_complaint(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0015")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    due_at = datetime.now(timezone.utc) + timedelta(days=2)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": due_at.isoformat(),
            "escalation_state": "Escalated",
            "escalation_reason": (
                "High-priority plumbing issue requires follow-up."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["complaint_reference"] == complaint.complaint_reference
    assert payload["escalation_state"] == "Escalated"
    assert payload["escalation_reason"] == (
        "High-priority plumbing issue requires follow-up."
    )
    assert payload["set_by_user_id"] == staff_user.id
    assert payload["is_overdue"] is False

    escalation = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .one()
    )

    assert escalation.escalation_state == "Escalated"
    assert escalation.escalation_reason == (
        "High-priority plumbing issue requires follow-up."
    )
    assert escalation.set_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "CREATE_COMPLAINT_ESCALATION",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_staff_can_update_existing_escalation(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0016")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    initial_due_at = datetime.now(timezone.utc) + timedelta(days=1)

    initial_escalation = ComplaintEscalation(
        complaint_id=complaint.id,
        due_at=initial_due_at,
        escalation_state="OnTrack",
        escalation_reason="Initial due date set for monitoring.",
        set_by_user_id=staff_user.id,
    )

    db_session.add(initial_escalation)
    db_session.commit()
    db_session.refresh(initial_escalation)

    updated_due_at = datetime.now(timezone.utc) + timedelta(days=3)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": updated_due_at.isoformat(),
            "escalation_state": "Escalated",
            "escalation_reason": (
                "Deadline extended after maintenance assessment."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["escalation_state"] == "Escalated"
    assert payload["escalation_reason"] == (
        "Deadline extended after maintenance assessment."
    )
    assert payload["set_by_user_id"] == staff_user.id
    assert payload["is_overdue"] is False

    escalations = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .all()
    )

    assert len(escalations) == 1

    escalation = escalations[0]

    assert escalation.escalation_state == "Escalated"
    assert escalation.escalation_reason == (
        "Deadline extended after maintenance assessment."
    )
    assert escalation.set_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "UPDATE_COMPLAINT_ESCALATION",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_escalated_complaint_requires_reason(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0017")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    due_at = datetime.now(timezone.utc) + timedelta(days=2)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": due_at.isoformat(),
            "escalation_state": "Escalated",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "escalation_reason is required when escalation_state is Escalated."
    )

    escalation_count = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert escalation_count == 0
    assert audit_count == 0

def test_escalated_complaint_requires_future_due_date(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0018")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "escalation_state": "Escalated",
            "escalation_reason": (
                "Urgent issue requires a formal escalation deadline."
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "due_at is required when escalation_state is Escalated."
    )

    escalation_count = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert escalation_count == 0
    assert audit_count == 0

def test_escalation_rejects_due_date_in_the_past(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0019")

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    past_due_at = datetime.now(timezone.utc) - timedelta(minutes=5)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": past_due_at.isoformat(),
            "escalation_state": "Escalated",
            "escalation_reason": (
                "This payload must be rejected because its deadline "
                "is in the past."
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "due_at must be in the future."

    escalation_count = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert escalation_count == 0
    assert audit_count == 0

def test_resolved_complaint_cannot_be_escalated(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0020")
    complaint.status = "Resolved"

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    due_at = datetime.now(timezone.utc) + timedelta(days=2)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": due_at.isoformat(),
            "escalation_state": "Escalated",
            "escalation_reason": (
                "A resolved complaint must not accept escalation changes."
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Escalation cannot be created or updated for a Resolved or "
        "Closed complaint."
    )

    escalation_count = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert escalation_count == 0
    assert audit_count == 0

def test_staff_cannot_escalate_complaint_assigned_to_another_department(
    client: TestClient,
    db_session: Session,
    other_department: Department,
    other_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0021")
    complaint.assigned_department = other_department.name

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_to_user_id=other_department_staff.id,
        assigned_department_id=other_department.id,
        assigned_by_user_id=other_department_staff.id,
        assignment_note="Assigned to electrical maintenance.",
    )

    db_session.add(assignment)
    db_session.commit()

    due_at = datetime.now(timezone.utc) + timedelta(days=2)

    response = client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": due_at.isoformat(),
            "escalation_state": "Escalated",
            "escalation_reason": (
                "Cross-department escalation attempt must be rejected."
            ),
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "You do not have access to this complaint."
    )

    escalation_count = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert escalation_count == 0
    assert audit_count == 0

def test_admin_can_escalate_complaint_assigned_to_another_department(
    admin_client: TestClient,
    db_session: Session,
    admin_user: User,
    other_department: Department,
    other_department_staff: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0022")
    complaint.assigned_department = other_department.name

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    assignment = ComplaintAssignment(
        complaint_id=complaint.id,
        assigned_to_user_id=other_department_staff.id,
        assigned_department_id=other_department.id,
        assigned_by_user_id=admin_user.id,
        assignment_note="Assigned to electrical maintenance.",
    )

    db_session.add(assignment)
    db_session.commit()

    due_at = datetime.now(timezone.utc) + timedelta(days=3)

    response = admin_client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": due_at.isoformat(),
            "escalation_state": "Escalated",
            "escalation_reason": (
                "Admin escalation for cross-department operational follow-up."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["escalation_state"] == "Escalated"
    assert payload["escalation_reason"] == (
        "Admin escalation for cross-department operational follow-up."
    )
    assert payload["set_by_user_id"] == admin_user.id
    assert payload["is_overdue"] is False

    escalation = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .one()
    )

    assert escalation.escalation_state == "Escalated"
    assert escalation.escalation_reason == (
        "Admin escalation for cross-department operational follow-up."
    )
    assert escalation.set_by_user_id == admin_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "CREATE_COMPLAINT_ESCALATION",
        )
        .one()
    )

    assert audit_log.actor_user_id == admin_user.id

def test_student_cannot_escalate_own_complaint(
    student_client: TestClient,
    db_session: Session,
    student_user: User,
) -> None:
    complaint = create_owned_complaint(
        db_session=db_session,
        complaint_reference="CR-TEST-0023",
        owner=student_user,
    )

    due_at = datetime.now(timezone.utc) + timedelta(days=2)

    response = student_client.put(
        f"/complaints/{complaint.complaint_reference}/escalation",
        json={
            "due_at": due_at.isoformat(),
            "escalation_state": "Escalated",
            "escalation_reason": (
                "Students cannot create operational escalation records."
            ),
        },
    )

    assert response.status_code == 403

    db_session.refresh(complaint)

    assert complaint.status == "Open"

    escalation_count = (
        db_session.query(ComplaintEscalation)
        .filter(ComplaintEscalation.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert escalation_count == 0
    assert audit_count == 0

def test_staff_can_verify_impact_with_reported_population_default(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0024")
    complaint.affected_population = 37

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": (
                "Confirmed reported impact during maintenance inspection."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["complaint_reference"] == complaint.complaint_reference
    assert payload["reported_affected_population"] == 37
    assert payload["verified_affected_population"] == 37
    assert payload["impact_verification_status"] == "Verified"
    assert payload["impact_verification_note"] == (
        "Confirmed reported impact during maintenance inspection."
    )
    assert payload["verified_by_user_id"] == staff_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.reported_affected_population == 37
    assert verification.verified_affected_population == 37
    assert verification.impact_verification_status == "Verified"
    assert verification.verified_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_adjusted_impact_verification_requires_verified_population(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0025")
    complaint.affected_population = 40

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Adjusted",
            "impact_verification_note": (
                "Inspection found a different affected population."
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "verified_affected_population is required when "
        "impact_verification_status is Adjusted."
    )

    verification_count = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .count()
    )

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
        )
        .count()
    )

    assert verification_count == 0
    assert audit_count == 0

def test_staff_can_adjust_verified_affected_population(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0026")
    complaint.affected_population = 40

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Adjusted",
            "verified_affected_population": 28,
            "impact_verification_note": (
                "Inspection confirmed that 28 people were affected."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["reported_affected_population"] == 40
    assert payload["verified_affected_population"] == 28
    assert payload["impact_verification_status"] == "Adjusted"
    assert payload["impact_verification_note"] == (
        "Inspection confirmed that 28 people were affected."
    )
    assert payload["verified_by_user_id"] == staff_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.reported_affected_population == 40
    assert verification.verified_affected_population == 28
    assert verification.impact_verification_status == "Adjusted"
    assert verification.verified_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_rejected_impact_verification_clears_verified_population(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0027")
    complaint.affected_population = 30

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Rejected",
            "verified_affected_population": 99,
            "impact_verification_note": (
                "Reported impact could not be confirmed during inspection."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["reported_affected_population"] == 30
    assert payload["verified_affected_population"] is None
    assert payload["impact_verification_status"] == "Rejected"
    assert payload["impact_verification_note"] == (
        "Reported impact could not be confirmed during inspection."
    )
    assert payload["verified_by_user_id"] == staff_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.reported_affected_population == 30
    assert verification.verified_affected_population is None
    assert verification.impact_verification_status == "Rejected"
    assert verification.verified_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_cannot_verify_complaint_impact_twice(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0028")
    complaint.affected_population = 18

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    first_response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": "Initial field verification completed.",
        },
    )

    assert first_response.status_code == 200

    second_response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Adjusted",
            "verified_affected_population": 12,
            "impact_verification_note": "Attempted replacement verification.",
        },
    )

    assert second_response.status_code == 400
    assert second_response.json()["detail"] == (
        "Complaint impact has already been verified."
    )

    verifications = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .all()
    )

    assert len(verifications) == 1

    verification = verifications[0]
    assert verification.impact_verification_status == "Verified"
    assert verification.reported_affected_population == 18
    assert verification.verified_affected_population == 18
    assert verification.impact_verification_note == (
        "Initial field verification completed."
    )
    assert verification.verified_by_user_id == staff_user.id

    audit_logs = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .all()
    )

    assert len(audit_logs) == 1
    assert audit_logs[0].actor_user_id == staff_user.id

def test_impact_verification_returns_404_for_missing_complaint(
    client: TestClient,
    db_session: Session,
) -> None:
    response = client.patch(
        "/complaints/CR-DOES-NOT-EXIST/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": "Attempted verification.",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Complaint not found: CR-DOES-NOT-EXIST"
    )

    assert db_session.query(ComplaintVerification).count() == 0

    audit_count = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "VERIFY_COMPLAINT_IMPACT")
        .count()
    )

    assert audit_count == 0

def test_non_staff_cannot_verify_complaint_impact(
    student_client: TestClient,
    db_session: Session,
    student_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0029")
    complaint.affected_population = 24

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = student_client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": (
                "Unauthorized user attempted impact verification."
            ),
        },
    )

    assert response.status_code == 403

    verification_count = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .count()
    )

    assert verification_count == 0

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .count()
    )

    assert audit_count == 0

def test_admin_can_verify_complaint_impact(
    admin_client: TestClient,
    db_session: Session,
    admin_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0030")
    complaint.affected_population = 16

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = admin_client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": (
                "Administrator confirmed the reported impact."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["complaint_reference"] == complaint.complaint_reference
    assert payload["reported_affected_population"] == 16
    assert payload["verified_affected_population"] == 16
    assert payload["impact_verification_status"] == "Verified"
    assert payload["impact_verification_note"] == (
        "Administrator confirmed the reported impact."
    )
    assert payload["verified_by_user_id"] == admin_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.reported_affected_population == 16
    assert verification.verified_affected_population == 16
    assert verification.impact_verification_status == "Verified"
    assert verification.verified_by_user_id == admin_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == admin_user.id

def test_impact_verification_rejects_invalid_status(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0031")
    complaint.affected_population = 22

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "PendingReview",
            "impact_verification_note": (
                "This status is not part of the allowed workflow."
            ),
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "impact_verification_status"]
            for error in errors
    )

    verification_count = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .count()
    )

    assert verification_count == 0

    audit_count = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .count()
    )

    assert audit_count == 0

def test_impact_verification_allows_omitted_note(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0032")
    complaint.affected_population = 14

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["complaint_reference"] == complaint.complaint_reference
    assert payload["reported_affected_population"] == 14
    assert payload["verified_affected_population"] == 14
    assert payload["impact_verification_status"] == "Verified"
    assert payload["impact_verification_note"] is None
    assert payload["verified_by_user_id"] == staff_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.reported_affected_population == 14
    assert verification.verified_affected_population == 14
    assert verification.impact_verification_status == "Verified"
    assert verification.impact_verification_note is None
    assert verification.verified_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_impact_verification_normalizes_whitespace_only_note_to_none(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0033")
    complaint.affected_population = 11

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": "   \n\t  ",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["impact_verification_status"] == "Verified"
    assert payload["verified_affected_population"] == 11
    assert payload["impact_verification_note"] is None

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.impact_verification_note is None

def test_impact_verification_trims_note_whitespace(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0034")
    complaint.affected_population = 9

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": (
                "  Field inspection confirmed the reported impact.  "
            ),
        },
    )

    assert response.status_code == 200

    expected_note = "Field inspection confirmed the reported impact."

    payload = response.json()

    assert payload["impact_verification_status"] == "Verified"
    assert payload["verified_affected_population"] == 9
    assert payload["impact_verification_note"] == expected_note

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.impact_verification_note == expected_note

def test_impact_verification_rejects_negative_verified_population(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0035")
    complaint.affected_population = 10

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Adjusted",
            "verified_affected_population": -1,
            "impact_verification_note": (
                "Negative values must not be accepted."
            ),
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "verified_affected_population"]
        for error in errors
    )

    assert (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .count()
    ) == 0

    assert (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .count()
    ) == 0

def test_impact_verification_rejects_verified_population_above_maximum(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0036")
    complaint.affected_population = 10

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Adjusted",
            "verified_affected_population": 100001,
            "impact_verification_note": (
                "Values above the maximum must not be accepted."
            ),
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "verified_affected_population"]
        for error in errors
    )

    assert (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .count()
    ) == 0

    assert (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .count()
    ) == 0

def test_adjusted_impact_verification_allows_zero_population(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0037")
    complaint.affected_population = 10

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Adjusted",
            "verified_affected_population": 0,
            "impact_verification_note": (
                "Inspection found no confirmed affected individuals."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["reported_affected_population"] == 10
    assert payload["verified_affected_population"] == 0
    assert payload["impact_verification_status"] == "Adjusted"
    assert payload["verified_by_user_id"] == staff_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.reported_affected_population == 10
    assert verification.verified_affected_population == 0
    assert verification.impact_verification_status == "Adjusted"
    assert verification.verified_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_adjusted_impact_verification_allows_maximum_population(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0038")
    complaint.affected_population = 10

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Adjusted",
            "verified_affected_population": 100000,
            "impact_verification_note": (
                "Maximum allowed verified population accepted."
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["reported_affected_population"] == 10
    assert payload["verified_affected_population"] == 100000
    assert payload["impact_verification_status"] == "Adjusted"
    assert payload["verified_by_user_id"] == staff_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.reported_affected_population == 10
    assert verification.verified_affected_population == 100000
    assert verification.impact_verification_status == "Adjusted"
    assert verification.verified_by_user_id == staff_user.id

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id

def test_impact_verification_rejects_note_over_maximum_length(
    client: TestClient,
    db_session: Session,
) -> None:
    complaint = make_test_complaint("CR-TEST-0039")
    complaint.affected_population = 5

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": "x" * 1001,
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "impact_verification_note"]
        for error in errors
    )

    assert (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .count()
    ) == 0

    assert (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .count()
    ) == 0

def test_impact_verification_allows_note_at_maximum_length(
    client: TestClient,
    db_session: Session,
    staff_user: User,
) -> None:
    complaint = make_test_complaint("CR-TEST-0040")
    complaint.affected_population = 5

    db_session.add(complaint)
    db_session.commit()
    db_session.refresh(complaint)

    note = "x" * 1000

    response = client.patch(
        f"/complaints/{complaint.complaint_reference}/impact-verification",
        json={
            "impact_verification_status": "Verified",
            "impact_verification_note": note,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["impact_verification_status"] == "Verified"
    assert payload["verified_affected_population"] == 5
    assert payload["impact_verification_note"] == note
    assert payload["verified_by_user_id"] == staff_user.id

    verification = (
        db_session.query(ComplaintVerification)
        .filter(ComplaintVerification.complaint_id == complaint.id)
        .one()
    )

    assert verification.impact_verification_note == note
    assert len(verification.impact_verification_note) == 1000

    audit_log = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.entity_type == "Complaint",
            AuditLog.entity_id == str(complaint.id),
            AuditLog.action == "VERIFY_COMPLAINT_IMPACT",
        )
        .one()
    )

    assert audit_log.actor_user_id == staff_user.id