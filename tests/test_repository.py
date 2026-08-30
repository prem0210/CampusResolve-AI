from pathlib import Path

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
    )

    return session_factory()


def make_complaint(reference: str, priority: str, status: str) -> Complaint:
    return Complaint(
        complaint_reference=reference,
        complaint_text="Synthetic test complaint",
        language="en",
        location_type="Hostel",
        specific_location="Hostel Block A",
        affected_population=20,
        safety_flag=False,
        repeat_count=0,
        predicted_category="Water and Plumbing",
        category_confidence=0.90,
        assigned_department="Plumbing and Civil Maintenance",
        predicted_priority=priority,
        priority_confidence=0.85,
        estimated_resolution_hours=12.0,
        prediction_interval_plus_minus_hours=6.0,
        duplicate_threshold=0.70,
        possible_duplicate=False,
        status=status,
    )


def test_complaint_reference_increments_by_day(tmp_path: Path) -> None:
    db = make_test_session(tmp_path)

    try:
        first_reference = build_complaint_reference(db)
        assert first_reference.endswith("-0001")

        db.add(make_complaint(first_reference, "High", "Open"))
        db.commit()

        second_reference = build_complaint_reference(db)
        assert second_reference.endswith("-0002")
    finally:
        db.close()


def test_list_complaints_filters_by_status_and_priority(tmp_path: Path) -> None:
    db = make_test_session(tmp_path)

    try:
        db.add_all(
            [
                make_complaint("CR-TEST-0001", "Critical", "Open"),
                make_complaint("CR-TEST-0002", "High", "In Progress"),
                make_complaint("CR-TEST-0003", "Low", "Closed"),
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


def test_update_complaint_and_dashboard_summary(tmp_path: Path) -> None:
    db = make_test_session(tmp_path)

    try:
        complaint = make_complaint("CR-TEST-0001", "Critical", "Open")
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

        summary = get_dashboard_summary(db)

        assert summary["total_complaints"] == 1
        assert summary["in_progress_complaints"] == 1
        assert summary["critical_open_complaints"] == 1
    finally:
        db.close()