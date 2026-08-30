from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import Complaint
from sqlalchemy import and_, desc

def build_complaint_reference(db: Session) -> str:
    date_prefix = datetime.now().strftime("%Y%m%d")
    prefix = f"CR-{date_prefix}-"

    last_reference = db.scalar(
        select(Complaint.complaint_reference)
        .where(Complaint.complaint_reference.like(f"{prefix}%"))
        .order_by(Complaint.complaint_reference.desc())
        .limit(1)
    )

    if last_reference:
        sequence = int(last_reference.rsplit("-", maxsplit=1)[1]) + 1
    else:
        sequence = 1

    return f"{prefix}{sequence:04d}"


def create_complaint(
    db: Session,
    payload: dict[str, Any],
    prediction: dict[str, Any],
) -> Complaint:
    duplicate_candidates = prediction.get("duplicate_candidates", [])

    top_duplicate_id = None
    top_duplicate_similarity = None

    if duplicate_candidates:
        top_duplicate_id = duplicate_candidates[0]["complaint_id"]
        top_duplicate_similarity = duplicate_candidates[0]["similarity_score"]

    complaint = Complaint(
        complaint_reference=build_complaint_reference(db),
        complaint_text=payload["complaint_text"],
        language=payload["language"],
        location_type=payload["location_type"],
        specific_location=payload["specific_location"],
        affected_population=payload["affected_population"],
        safety_flag=bool(payload["safety_flag"]),
        repeat_count=payload["repeat_count"],
        predicted_category=prediction["predicted_category"],
        category_confidence=prediction["category_confidence"],
        assigned_department=prediction["assigned_department"],
        predicted_priority=prediction["predicted_priority"],
        priority_confidence=prediction["priority_confidence"],
        estimated_resolution_hours=prediction["estimated_resolution_hours"],
        prediction_interval_plus_minus_hours=prediction[
            "prediction_interval_plus_minus_hours"
        ],
        duplicate_threshold=prediction["duplicate_threshold"],
        possible_duplicate=prediction["possible_duplicate"],
        top_duplicate_id=top_duplicate_id,
        top_duplicate_similarity=top_duplicate_similarity,
        status="Open",
    )

    db.add(complaint)
    db.commit()
    db.refresh(complaint)

    return complaint


def count_complaints(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(Complaint)) or 0)



def get_complaint_by_reference(
    db: Session,
    complaint_reference: str,
) -> Complaint | None:
    return db.scalar(
        select(Complaint).where(
            Complaint.complaint_reference == complaint_reference
        )
    )


def list_complaints(
    db: Session,
    status: str | None = None,
    priority: str | None = None,
    department: str | None = None,
    category: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[Complaint]]:
    filters = []

    if status:
        filters.append(Complaint.status == status)

    if priority:
        filters.append(Complaint.predicted_priority == priority)

    if department:
        filters.append(Complaint.assigned_department == department)

    if category:
        filters.append(Complaint.predicted_category == category)

    query = select(Complaint)

    if filters:
        query = query.where(and_(*filters))

    total = int(
        db.scalar(
            select(func.count())
            .select_from(Complaint)
            .where(and_(*filters))
            if filters
            else select(func.count()).select_from(Complaint)
        )
        or 0
    )

    complaints = list(
        db.scalars(
            query.order_by(desc(Complaint.created_at))
            .offset(offset)
            .limit(limit)
        ).all()
    )

    return total, complaints


def update_complaint(
    db: Session,
    complaint: Complaint,
    updates: dict[str, Any],
) -> Complaint:
    for field_name, value in updates.items():
        setattr(complaint, field_name, value)

    db.commit()
    db.refresh(complaint)

    return complaint


def get_dashboard_summary(db: Session) -> dict[str, int]:
    def count_with_filters(*conditions: Any) -> int:
        query = select(func.count()).select_from(Complaint)

        if conditions:
            query = query.where(and_(*conditions))

        return int(db.scalar(query) or 0)

    return {
        "total_complaints": count_with_filters(),
        "open_complaints": count_with_filters(Complaint.status == "Open"),
        "in_progress_complaints": count_with_filters(
            Complaint.status == "In Progress"
        ),
        "resolved_complaints": count_with_filters(
            Complaint.status == "Resolved"
        ),
        "closed_complaints": count_with_filters(Complaint.status == "Closed"),
        "critical_open_complaints": count_with_filters(
            Complaint.status.in_(["Open", "In Progress"]),
            Complaint.predicted_priority == "Critical",
        ),
        "high_open_complaints": count_with_filters(
            Complaint.status.in_(["Open", "In Progress"]),
            Complaint.predicted_priority == "High",
        ),
        "possible_duplicate_complaints": count_with_filters(
            Complaint.possible_duplicate.is_(True),
        ),
    }