from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.database import Base
from src.database.models import Complaint
from src.database.repository import (
    build_complaint_reference,
    get_dashboard_summary,
    list_complaints,
    update_complaint,
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