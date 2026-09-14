from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sqlalchemy import and_, desc

from src.database.models import (
    AuditLog,
    CampusBlock,
    Complaint,
    ComplaintOwnership,
    ComplaintStatusHistory,
    ComplaintVerification,
    Department,
    LocationType,
    MLFeedbackRecord,
    User,
)

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

def list_departments(
    db: Session,
    active_only: bool = True,
) -> list[Department]:
    query = select(Department).order_by(Department.name)

    if active_only:
        query = query.where(Department.is_active.is_(True))

    return list(db.scalars(query).all())


def list_location_types(
    db: Session,
    active_only: bool = True,
) -> list[LocationType]:
    query = select(LocationType).order_by(LocationType.name)

    if active_only:
        query = query.where(LocationType.is_active.is_(True))

    return list(db.scalars(query).all())


def list_campus_blocks(
    db: Session,
    location_type_id: int | None = None,
    active_only: bool = True,
) -> list[CampusBlock]:
    query = select(CampusBlock).order_by(CampusBlock.name)

    filters = []

    if location_type_id is not None:
        filters.append(CampusBlock.location_type_id == location_type_id)

    if active_only:
        filters.append(CampusBlock.is_active.is_(True))

    if filters:
        query = query.where(and_(*filters))

    return list(db.scalars(query).all())

def get_user_by_email(
    db: Session,
    email: str,
) -> User | None:
    return db.scalar(
        select(User).where(User.email == email.lower())
    )


def get_user_by_id(
    db: Session,
    user_id: int,
) -> User | None:
    return db.get(User, user_id)

def get_department_by_id(
    db: Session,
    department_id: int,
) -> Department | None:
    return db.get(Department, department_id)


def get_department_by_code(
    db: Session,
    code: str,
) -> Department | None:
    return db.scalar(
        select(Department).where(Department.code == code.upper())
    )


def get_department_by_name(
    db: Session,
    name: str,
) -> Department | None:
    return db.scalar(
        select(Department).where(Department.name == name.strip())
    )


def create_department(
    db: Session,
    data: dict[str, Any],
) -> Department:
    department = Department(
        code=str(data["code"]).strip().upper(),
        name=str(data["name"]).strip(),
        description=(
            str(data["description"]).strip()
            if data.get("description")
            else None
        ),
        contact_email=(
            str(data["contact_email"]).strip().lower()
            if data.get("contact_email")
            else None
        ),
    )

    db.add(department)
    db.commit()
    db.refresh(department)

    return department


def get_campus_block_by_id(
    db: Session,
    block_id: int,
) -> CampusBlock | None:
    return db.get(CampusBlock, block_id)


def get_campus_block_by_code(
    db: Session,
    code: str,
) -> CampusBlock | None:
    return db.scalar(
        select(CampusBlock).where(CampusBlock.code == code.upper())
    )


def get_campus_block_by_name(
    db: Session,
    name: str,
) -> CampusBlock | None:
    return db.scalar(
        select(CampusBlock).where(CampusBlock.name == name.strip())
    )


def get_location_type_by_id(
    db: Session,
    location_type_id: int,
) -> LocationType | None:
    return db.get(LocationType, location_type_id)


def create_campus_block(
    db: Session,
    data: dict[str, Any],
) -> CampusBlock:
    campus_block = CampusBlock(
        code=str(data["code"]).strip().upper(),
        name=str(data["name"]).strip(),
        location_type_id=int(data["location_type_id"]),
        capacity=data.get("capacity"),
        responsible_department_id=data.get(
            "responsible_department_id"
        ),
    )

    db.add(campus_block)
    db.commit()
    db.refresh(campus_block)

    return campus_block

def update_department(
    db: Session,
    department: Department,
    updates: dict[str, Any],
) -> Department:
    for field_name, value in updates.items():
        if field_name == "name" and value is not None:
            value = str(value).strip()
        elif field_name == "description":
            value = str(value).strip() if value else None
        elif field_name == "contact_email":
            value = str(value).strip().lower() if value else None

        setattr(department, field_name, value)

    db.commit()
    db.refresh(department)

    return department


def update_campus_block(
    db: Session,
    campus_block: CampusBlock,
    updates: dict[str, Any],
) -> CampusBlock:
    for field_name, value in updates.items():
        if field_name == "name" and value is not None:
            value = str(value).strip()

        setattr(campus_block, field_name, value)

    db.commit()
    db.refresh(campus_block)

    return campus_block

def create_audit_log(
    db: Session,
    actor_user_id: int | None,
    action: str,
    entity_type: str,
    entity_id: str,
    details: str | None = None,
) -> AuditLog:
    audit_log = AuditLog(
        actor_user_id=actor_user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return audit_log


def list_audit_logs(
    db: Session,
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 100,
) -> list[AuditLog]:
    query = select(AuditLog).order_by(desc(AuditLog.created_at))

    filters = []

    if entity_type:
        filters.append(AuditLog.entity_type == entity_type)

    if entity_id:
        filters.append(AuditLog.entity_id == entity_id)

    if filters:
        query = query.where(and_(*filters))

    return list(db.scalars(query.limit(limit)).all())

def get_complaint_verification(
    db: Session,
    complaint_id: int,
) -> ComplaintVerification | None:
    return db.scalar(
        select(ComplaintVerification).where(
            ComplaintVerification.complaint_id == complaint_id
        )
    )


def get_or_create_complaint_verification(
    db: Session,
    complaint: Complaint,
) -> ComplaintVerification:
    verification = get_complaint_verification(
        db=db,
        complaint_id=complaint.id,
    )

    if verification is None:
        verification = ComplaintVerification(
            complaint_id=complaint.id,
            reported_affected_population=complaint.affected_population,
            impact_verification_status="Unverified",
        )
        db.add(verification)
        db.commit()
        db.refresh(verification)

    return verification


def update_complaint_verification(
    db: Session,
    verification: ComplaintVerification,
    updates: dict[str, Any],
    verified_by_user_id: int,
) -> ComplaintVerification:
    for field_name, value in updates.items():
        if field_name == "impact_verification_note":
            value = str(value).strip() if value else None

        setattr(verification, field_name, value)

    verification.verified_by_user_id = verified_by_user_id
    verification.verified_at = datetime.utcnow()

    db.commit()
    db.refresh(verification)

    return verification

def create_status_history(
    db: Session,
    complaint_id: int,
    old_status: str | None,
    new_status: str,
    changed_by_user_id: int | None,
    note: str | None = None,
) -> ComplaintStatusHistory:
    history = ComplaintStatusHistory(
        complaint_id=complaint_id,
        old_status=old_status,
        new_status=new_status,
        changed_by_user_id=changed_by_user_id,
        note=note.strip() if note else None,
    )

    db.add(history)
    db.commit()
    db.refresh(history)

    return history


def list_status_history(
    db: Session,
    complaint_id: int,
) -> list[ComplaintStatusHistory]:
    query = (
        select(ComplaintStatusHistory)
        .where(ComplaintStatusHistory.complaint_id == complaint_id)
        .order_by(ComplaintStatusHistory.changed_at.asc())
    )

    return list(db.scalars(query).all())

def get_ml_feedback_record(
    db: Session,
    complaint_id: int,
) -> MLFeedbackRecord | None:
    return db.scalar(
        select(MLFeedbackRecord).where(
            MLFeedbackRecord.complaint_id == complaint_id
        )
    )


def get_or_create_ml_feedback_record(
    db: Session,
    complaint: Complaint,
) -> MLFeedbackRecord:
    feedback = get_ml_feedback_record(
        db=db,
        complaint_id=complaint.id,
    )

    if feedback is None:
        feedback = MLFeedbackRecord(
            complaint_id=complaint.id,
            training_eligible=False,
            duplicate_decision="NotReviewed",
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)

    return feedback


def update_ml_feedback_record(
    db: Session,
    feedback: MLFeedbackRecord,
    updates: dict[str, Any],
    reviewed_by_user_id: int,
) -> MLFeedbackRecord:
    for field_name, value in updates.items():
        if field_name == "exclusion_reason":
            value = str(value).strip() if value else None

        setattr(feedback, field_name, value)

    feedback.reviewed_by_user_id = reviewed_by_user_id
    feedback.reviewed_at = datetime.utcnow()

    db.commit()
    db.refresh(feedback)

    return feedback

def create_complaint_ownership(
    db: Session,
    complaint_id: int,
    submitted_by_user_id: int,
) -> ComplaintOwnership:
    ownership = ComplaintOwnership(
        complaint_id=complaint_id,
        submitted_by_user_id=submitted_by_user_id,
    )

    db.add(ownership)
    db.commit()
    db.refresh(ownership)

    return ownership


def get_complaint_ownership(
    db: Session,
    complaint_id: int,
) -> ComplaintOwnership | None:
    return db.scalar(
        select(ComplaintOwnership).where(
            ComplaintOwnership.complaint_id == complaint_id
        )
    )


def is_complaint_owner(
    db: Session,
    complaint_id: int,
    user_id: int,
) -> bool:
    ownership = get_complaint_ownership(
        db=db,
        complaint_id=complaint_id,
    )

    return (
        ownership is not None
        and ownership.submitted_by_user_id == user_id
    )