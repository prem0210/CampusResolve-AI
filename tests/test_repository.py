from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.database import Base
from src.database.models import (
    CampusBlock,
    Complaint,
    ComplaintAssignment,
    ComplaintOwnership,
    ComplaintVerification,
    Department,
    LocationType,
    MLFeedbackRecord,
    User,
)
from src.database.repository import (
    build_complaint_reference,
    can_staff_access_complaint,
    create_audit_log,
    create_complaint_assignment,
    create_complaint_ownership,
    get_dashboard_summary,
    get_or_create_complaint_verification,
    get_or_create_ml_feedback_record,
    is_complaint_owner,
    list_audit_logs,
    list_complaints,
    list_unowned_complaints,
    update_campus_block,
    update_complaint,
    update_complaint_verification,
    update_department,
    update_ml_feedback_record,
)


def make_test_session(tmp_path: Path) -> Session:
    database_path = tmp_path / "test_campusresolve.db"

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

    return session_factory()


def make_complaint(
    reference: str,
    priority: str = "Medium",
    status: str = "Open",
    *,
    complaint_text: str = "Synthetic test complaint",
    assigned_department: str = "Plumbing and Civil Maintenance",
    created_at: datetime | None = None,
) -> Complaint:
    complaint = Complaint(
        complaint_reference=reference,
        complaint_text=complaint_text,
        language="en",
        location_type="Hostel",
        specific_location="Hostel Block A",
        affected_population=20,
        safety_flag=False,
        repeat_count=0,
        predicted_category="Water and Plumbing",
        category_confidence=0.90,
        assigned_department=assigned_department,
        predicted_priority=priority,
        priority_confidence=0.85,
        estimated_resolution_hours=12.0,
        prediction_interval_plus_minus_hours=6.0,
        duplicate_threshold=0.70,
        possible_duplicate=False,
        status=status,
    )

    if created_at is not None:
        complaint.created_at = created_at

    return complaint

def make_department(
    code: str,
    name: str,
) -> Department:
    return Department(
        code=code,
        name=name,
        is_active=True,
    )


def make_staff_user(
    email: str,
    department_id: int,
) -> User:
    return User(
        full_name=email.split("@")[0].replace(".", " ").title(),
        email=email,
        password_hash="test-password-hash",
        role="Staff",
        department_id=department_id,
        is_active=True,
    )


def test_complaint_reference_increments_by_day(tmp_path: Path) -> None:
    db = make_test_session(tmp_path)

    try:
        first_reference = build_complaint_reference(db)

        assert first_reference.endswith("-0001")

        db.add(
            make_complaint(
                reference=first_reference,
                priority="High",
                status="Open",
            )
        )
        db.commit()

        second_reference = build_complaint_reference(db)

        assert second_reference.endswith("-0002")

    finally:
        db.close()


def test_list_complaints_filters_by_status_and_priority(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        db.add_all(
            [
                make_complaint(
                    "CR-TEST-0001",
                    priority="Critical",
                    status="Open",
                ),
                make_complaint(
                    "CR-TEST-0002",
                    priority="High",
                    status="In Progress",
                ),
                make_complaint(
                    "CR-TEST-0003",
                    priority="Low",
                    status="Closed",
                ),
            ]
        )
        db.commit()

        total, complaints = list_complaints(
            db=db,
            status="Open",
            priority="Critical",
        )

        assert total == 1
        assert len(complaints) == 1
        assert complaints[0].complaint_reference == "CR-TEST-0001"

    finally:
        db.close()


def test_list_complaints_without_filters_returns_all(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        db.add_all(
            [
                make_complaint("CR-TEST-0001"),
                make_complaint("CR-TEST-0002"),
                make_complaint("CR-TEST-0003"),
            ]
        )
        db.commit()

        total, complaints = list_complaints(db=db)

        assert total == 3
        assert len(complaints) == 3

    finally:
        db.close()


def test_list_complaints_returns_empty_for_no_match(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        db.add(
            make_complaint(
                "CR-TEST-0001",
                priority="Low",
                status="Closed",
            )
        )
        db.commit()

        total, complaints = list_complaints(
            db=db,
            status="Open",
            priority="Critical",
        )

        assert total == 0
        assert complaints == []

    finally:
        db.close()


def test_list_complaints_supports_pagination(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        db.add_all(
            [
                make_complaint("CR-TEST-0001"),
                make_complaint("CR-TEST-0002"),
                make_complaint("CR-TEST-0003"),
            ]
        )
        db.commit()

        total, complaints = list_complaints(
            db=db,
            limit=1,
            offset=1,
        )

        assert total == 3
        assert len(complaints) == 1

    finally:
        db.close()


def test_update_complaint_persists_allowed_fields(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        complaint = make_complaint(
            "CR-TEST-0001",
            priority="Critical",
            status="Open",
        )
        db.add(complaint)
        db.commit()
        db.refresh(complaint)

        updated = update_complaint(
            db=db,
            complaint=complaint,
            updates={
                "status": "In Progress",
                "staff_notes": "Technician assigned.",
            },
        )

        assert updated.status == "In Progress"
        assert updated.staff_notes == "Technician assigned."

        db.expire_all()

        persisted = db.get(Complaint, complaint.id)

        assert persisted is not None
        assert persisted.status == "In Progress"
        assert persisted.staff_notes == "Technician assigned."

    finally:
        db.close()


def test_update_complaint_rejects_unknown_field(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        complaint = make_complaint("CR-TEST-0001")
        db.add(complaint)
        db.commit()
        db.refresh(complaint)

        with pytest.raises(ValueError):
            update_complaint(
                db=db,
                complaint=complaint,
                updates={"not_a_complaint_field": "invalid"},
            )

    finally:
        db.close()

def test_update_complaint_rejects_non_model_field(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        complaint = make_complaint("CR-TEST-0001")
        db.add(complaint)
        db.commit()
        db.refresh(complaint)

        with pytest.raises(
            ValueError,
            match="Unsupported complaint update fields",
        ):
            update_complaint(
                db=db,
                complaint=complaint,
                updates={"resolution_notes": "Not a model field."},
            )

        db.refresh(complaint)

        assert not hasattr(complaint, "resolution_notes")

    finally:
        db.close()


def test_dashboard_summary_reflects_current_statuses(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        db.add_all(
            [
                make_complaint(
                    "CR-TEST-0001",
                    priority="Critical",
                    status="Open",
                ),
                make_complaint(
                    "CR-TEST-0002",
                    priority="High",
                    status="In Progress",
                ),
                make_complaint(
                    "CR-TEST-0003",
                    priority="Low",
                    status="Closed",
                ),
            ]
        )
        db.commit()

        summary = get_dashboard_summary(db)

        assert summary["total_complaints"] == 3
        assert summary["in_progress_complaints"] == 1
        assert summary["critical_open_complaints"] == 1

    finally:
        db.close()
    
def test_list_complaints_rejects_invalid_pagination(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        with pytest.raises(ValueError, match="limit must be at least 1"):
            list_complaints(
                db=db,
                limit=0,
                offset=0,
            )

        with pytest.raises(ValueError, match="offset cannot be negative"):
            list_complaints(
                db=db,
                limit=10,
                offset=-1,
            )

    finally:
        db.close()


def test_list_complaints_rejects_excessive_limit(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        with pytest.raises(ValueError, match="limit cannot exceed 200"):
            list_complaints(
                db=db,
                limit=201,
                offset=0,
            )

    finally:
        db.close()

def test_list_complaints_uses_id_as_timestamp_tie_breaker(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        first = make_complaint("CR-TEST-0001")
        second = make_complaint("CR-TEST-0002")

        db.add_all([first, second])
        db.commit()

        same_created_at = first.created_at

        second.created_at = same_created_at
        db.commit()

        total, complaints = list_complaints(
            db=db,
            limit=100,
            offset=0,
        )

        assert total == 2
        assert [
            complaint.complaint_reference
            for complaint in complaints
        ] == [
            "CR-TEST-0002",
            "CR-TEST-0001",
        ]

    finally:
        db.close()

def test_other_paginated_lists_reject_invalid_pagination(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        with pytest.raises(ValueError, match="limit must be at least 1"):
            list_audit_logs(
                db=db,
                limit=0,
                offset=0,
            )

        with pytest.raises(ValueError, match="offset cannot be negative"):
            list_unowned_complaints(
                db=db,
                limit=10,
                offset=-1,
            )

    finally:
        db.close()

def test_staff_access_matches_direct_and_department_assignments(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        maintenance = make_department(
            code="MAINT",
            name="Maintenance Department",
        )
        information_technology = make_department(
            code="IT",
            name="Information Technology Department",
        )
        db.add_all([maintenance, information_technology])
        db.commit()
        db.refresh(maintenance)
        db.refresh(information_technology)

        maintenance_staff = make_staff_user(
            email="maintenance.staff@example.com",
            department_id=maintenance.id,
        )
        other_maintenance_staff = make_staff_user(
            email="maintenance.colleague@example.com",
            department_id=maintenance.id,
        )
        it_staff = make_staff_user(
            email="it.staff@example.com",
            department_id=information_technology.id,
        )
        admin = User(
            full_name="Test Administrator",
            email="admin@example.com",
            password_hash="test-password-hash",
            role="Admin",
            department_id=None,
            is_active=True,
        )
        db.add_all(
            [
                maintenance_staff,
                other_maintenance_staff,
                it_staff,
                admin,
            ]
        )
        db.commit()
        db.refresh(maintenance_staff)
        db.refresh(other_maintenance_staff)
        db.refresh(it_staff)
        db.refresh(admin)

        unassigned = make_complaint("CR-TEST-0001")
        directly_assigned = make_complaint("CR-TEST-0002")
        department_assigned = make_complaint("CR-TEST-0003")
        other_department_assigned = make_complaint("CR-TEST-0004")

        db.add_all(
            [
                unassigned,
                directly_assigned,
                department_assigned,
                other_department_assigned,
            ]
        )
        db.commit()

        for complaint in [
            unassigned,
            directly_assigned,
            department_assigned,
            other_department_assigned,
        ]:
            db.refresh(complaint)

        create_complaint_assignment(
            db=db,
            complaint_id=directly_assigned.id,
            assigned_to_user_id=maintenance_staff.id,
            assigned_department_id=information_technology.id,
            assignment_note="Direct staff assignment.",
            assigned_by_user_id=admin.id,
        )

        create_complaint_assignment(
            db=db,
            complaint_id=department_assigned.id,
            assigned_to_user_id=it_staff.id,
            assigned_department_id=maintenance.id,
            assignment_note="Maintenance department assignment.",
            assigned_by_user_id=admin.id,
        )

        create_complaint_assignment(
            db=db,
            complaint_id=other_department_assigned.id,
            assigned_to_user_id=it_staff.id,
            assigned_department_id=information_technology.id,
            assignment_note="IT department assignment.",
            assigned_by_user_id=admin.id,
        )

        assert can_staff_access_complaint(
            db=db,
            complaint_id=unassigned.id,
            staff_user_id=maintenance_staff.id,
            staff_department_id=maintenance.id,
        )

        assert can_staff_access_complaint(
            db=db,
            complaint_id=directly_assigned.id,
            staff_user_id=maintenance_staff.id,
            staff_department_id=maintenance.id,
        )

        assert can_staff_access_complaint(
            db=db,
            complaint_id=department_assigned.id,
            staff_user_id=maintenance_staff.id,
            staff_department_id=maintenance.id,
        )

        assert not can_staff_access_complaint(
            db=db,
            complaint_id=other_department_assigned.id,
            staff_user_id=maintenance_staff.id,
            staff_department_id=maintenance.id,
        )

        total, complaints = list_complaints(
            db=db,
            staff_user_id=maintenance_staff.id,
            staff_department_id=maintenance.id,
            limit=100,
            offset=0,
        )

        visible_references = {
            complaint.complaint_reference
            for complaint in complaints
        }

        assert total == 3
        assert visible_references == {
            "CR-TEST-0001",
            "CR-TEST-0002",
            "CR-TEST-0003",
        }

    finally:
        db.close()
        
def test_student_can_only_access_owned_complaints(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        first_student = User(
            full_name="First Student",
            email="first.student@example.com",
            password_hash="test-password-hash",
            role="Student",
            department_id=None,
            is_active=True,
        )
        second_student = User(
            full_name="Second Student",
            email="second.student@example.com",
            password_hash="test-password-hash",
            role="Student",
            department_id=None,
            is_active=True,
        )
        db.add_all([first_student, second_student])
        db.commit()
        db.refresh(first_student)
        db.refresh(second_student)

        first_complaint = make_complaint("CR-TEST-0001")
        second_complaint = make_complaint("CR-TEST-0002")
        unowned_complaint = make_complaint("CR-TEST-0003")

        db.add_all(
            [
                first_complaint,
                second_complaint,
                unowned_complaint,
            ]
        )
        db.commit()

        for complaint in [
            first_complaint,
            second_complaint,
            unowned_complaint,
        ]:
            db.refresh(complaint)

        create_complaint_ownership(
            db=db,
            complaint_id=first_complaint.id,
            submitted_by_user_id=first_student.id,
        )
        create_complaint_ownership(
            db=db,
            complaint_id=second_complaint.id,
            submitted_by_user_id=second_student.id,
        )

        assert is_complaint_owner(
            db=db,
            complaint_id=first_complaint.id,
            user_id=first_student.id,
        )

        assert not is_complaint_owner(
            db=db,
            complaint_id=second_complaint.id,
            user_id=first_student.id,
        )

        assert not is_complaint_owner(
            db=db,
            complaint_id=unowned_complaint.id,
            user_id=first_student.id,
        )

        total, complaints = list_complaints(
            db=db,
            submitted_by_user_id=first_student.id,
            limit=100,
            offset=0,
        )

        assert total == 1
        assert [
            complaint.complaint_reference
            for complaint in complaints
        ] == ["CR-TEST-0001"]

    finally:
        db.close()

def test_rejected_impact_verification_clears_verified_population(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        staff_user = User(
            full_name="Test Staff",
            email="staff@example.com",
            password_hash="test-password-hash",
            role="Staff",
            department_id=None,
            is_active=True,
        )
        complaint = make_complaint("CR-TEST-0001")

        db.add_all([staff_user, complaint])
        db.commit()
        db.refresh(staff_user)
        db.refresh(complaint)

        verification = get_or_create_complaint_verification(
            db=db,
            complaint=complaint,
        )

        updated = update_complaint_verification(
            db=db,
            verification=verification,
            updates={
                "impact_verification_status": "Rejected",
                "verified_affected_population": None,
                "impact_verification_note": (
                    "Reported impact could not be verified."
                ),
            },
            verified_by_user_id=staff_user.id,
        )

        assert updated.impact_verification_status == "Rejected"
        assert updated.verified_affected_population is None
        assert (
            updated.impact_verification_note
            == "Reported impact could not be verified."
        )
        assert updated.verified_by_user_id == staff_user.id
        assert updated.verified_at is not None

    finally:
        db.close()

def test_update_department_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        department = make_department(
            code="MAINT",
            name="Maintenance Department",
        )
        db.add(department)
        db.commit()
        db.refresh(department)

        with pytest.raises(
            ValueError,
            match="Unsupported department update fields",
        ):
            update_department(
                db=db,
                department=department,
                updates={"code": "CHANGED"},
            )

        db.refresh(department)

        assert department.code == "MAINT"

    finally:
        db.close()

def test_update_campus_block_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        department = make_department(
            code="MAINT",
            name="Maintenance Department",
        )
        location_type = LocationType(
            name="Hostel",
            is_active=True,
        )
        db.add_all([department, location_type])
        db.commit()
        db.refresh(department)
        db.refresh(location_type)

        campus_block = CampusBlock(
            code="HB-A",
            name="Hostel Block A",
            location_type_id=location_type.id,
            capacity=240,
            responsible_department_id=department.id,
            is_active=True,
        )
        db.add(campus_block)
        db.commit()
        db.refresh(campus_block)

        with pytest.raises(
            ValueError,
            match="Unsupported campus-block update fields",
        ):
            update_campus_block(
                db=db,
                campus_block=campus_block,
                updates={"code": "HB-CHANGED"},
            )

        db.refresh(campus_block)

        assert campus_block.code == "HB-A"

    finally:
        db.close()

def test_update_verification_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        staff_user = User(
            full_name="Test Staff",
            email="staff@example.com",
            password_hash="test-password-hash",
            role="Staff",
            department_id=None,
            is_active=True,
        )
        complaint = make_complaint("CR-TEST-0001")

        db.add_all([staff_user, complaint])
        db.commit()
        db.refresh(staff_user)
        db.refresh(complaint)

        verification = get_or_create_complaint_verification(
            db=db,
            complaint=complaint,
        )

        with pytest.raises(
            ValueError,
            match="Unsupported complaint-verification update fields",
        ):
            update_complaint_verification(
                db=db,
                verification=verification,
                updates={"verified_by_user_id": 999},
                verified_by_user_id=staff_user.id,
            )

        db.refresh(verification)

        assert verification.verified_by_user_id is None
        assert verification.verified_at is None

    finally:
        db.close()

def test_update_ml_feedback_rejects_audit_field_changes(
    tmp_path: Path,
) -> None:
    db = make_test_session(tmp_path)

    try:
        staff_user = User(
            full_name="Test Staff",
            email="staff@example.com",
            password_hash="test-password-hash",
            role="Staff",
            department_id=None,
            is_active=True,
        )
        complaint = make_complaint("CR-TEST-0001")

        db.add_all([staff_user, complaint])
        db.commit()
        db.refresh(staff_user)
        db.refresh(complaint)

        feedback = get_or_create_ml_feedback_record(
            db=db,
            complaint=complaint,
        )

        with pytest.raises(
            ValueError,
            match="Unsupported ML-feedback update fields",
        ):
            update_ml_feedback_record(
                db=db,
                feedback=feedback,
                updates={"reviewed_by_user_id": 999},
                reviewed_by_user_id=staff_user.id,
            )

        db.refresh(feedback)

        assert feedback.reviewed_by_user_id is None
        assert feedback.reviewed_at is None

    finally:
        db.close()