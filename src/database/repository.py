from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session

from src.database.models import (
    AuditLog,
    CampusBlock,
    Complaint,
    ComplaintAssignment,
    ComplaintAssignmentHistory,
    ComplaintEscalation,
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
    submitted_by_user_id: int | None = None,
    staff_user_id: int | None = None,
    staff_department_id: int | None = None,
    assigned_to_user_id: int | None = None,
    assignment_state: str | None = None,
    assigned_department_id: int | None = None,
    escalation_state: str | None = None,
    due_before: datetime | None = None,
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

    if submitted_by_user_id is not None:
        filters.append(
            Complaint.id.in_(
                select(ComplaintOwnership.complaint_id).where(
                    ComplaintOwnership.submitted_by_user_id
                    == submitted_by_user_id
                )
            )
        )

    assignment_exists = (
        select(ComplaintAssignment.id)
        .where(ComplaintAssignment.complaint_id == Complaint.id)
        .exists()
    )

    # A staff queue must expose only unassigned complaints, complaints
    # directly assigned to the staff member, or complaints assigned to
    # the staff member's department.
    if staff_user_id is not None:
        staff_visibility_filters = [
            ~assignment_exists,
            Complaint.id.in_(
                select(ComplaintAssignment.complaint_id).where(
                    ComplaintAssignment.assigned_to_user_id
                    == staff_user_id
                )
            ),
        ]

        if staff_department_id is not None:
            staff_visibility_filters.append(
                Complaint.id.in_(
                    select(ComplaintAssignment.complaint_id).where(
                        ComplaintAssignment.assigned_department_id
                        == staff_department_id
                    )
                )
            )

        filters.append(or_(*staff_visibility_filters))

    if assignment_state == "Assigned":
        filters.append(assignment_exists)

    if assignment_state == "Unassigned":
        filters.append(~assignment_exists)

    if assigned_to_user_id is not None:
        filters.append(
            Complaint.id.in_(
                select(ComplaintAssignment.complaint_id).where(
                    ComplaintAssignment.assigned_to_user_id
                    == assigned_to_user_id
                )
            )
        )

    if assigned_department_id is not None:
        filters.append(
            Complaint.id.in_(
                select(ComplaintAssignment.complaint_id).where(
                    ComplaintAssignment.assigned_department_id
                    == assigned_department_id
                )
            )
        )

    active_complaint_filter = Complaint.status.not_in(
        ["Resolved", "Closed"]
    )

    if escalation_state == "OnTrack":
        filters.append(
            Complaint.id.in_(
                select(ComplaintEscalation.complaint_id).where(
                    ComplaintEscalation.escalation_state == "OnTrack"
                )
            )
        )

    if escalation_state == "Escalated":
        filters.append(
            Complaint.id.in_(
                select(ComplaintEscalation.complaint_id).where(
                    ComplaintEscalation.escalation_state == "Escalated"
                )
            )
        )

    if escalation_state == "Overdue":
        filters.append(
            Complaint.id.in_(
                select(ComplaintEscalation.complaint_id).where(
                    ComplaintEscalation.due_at.is_not(None),
                    ComplaintEscalation.due_at < datetime.utcnow(),
                    ComplaintEscalation.escalation_state != "Escalated",
                )
            )
        )
        filters.append(active_complaint_filter)

    if due_before is not None:
        filters.append(
            Complaint.id.in_(
                select(ComplaintEscalation.complaint_id).where(
                    ComplaintEscalation.due_at.is_not(None),
                    ComplaintEscalation.due_at <= due_before,
                )
            )
        )

    query = select(Complaint)
    count_query = select(func.count()).select_from(Complaint)

    if filters:
        filter_expression = and_(*filters)
        query = query.where(filter_expression)
        count_query = count_query.where(filter_expression)

    total = int(db.scalar(count_query) or 0)

    complaints = list(
        db.scalars(
            query.order_by(desc(Complaint.created_at))
            .offset(offset)
            .limit(limit)
        ).all()
    )

    return total, complaints


COMPLAINT_UPDATE_FIELDS = {
    "status",
    "staff_notes",
    "resolution_notes",
}


def update_complaint(
    db: Session,
    complaint: Complaint,
    updates: dict[str, Any],
) -> Complaint:
    unknown_fields = set(updates) - COMPLAINT_UPDATE_FIELDS

    if unknown_fields:
        unknown_text = ", ".join(sorted(unknown_fields))
        raise ValueError(
            f"Unsupported complaint update fields: {unknown_text}"
        )

    for field_name, value in updates.items():
        if field_name in {"staff_notes", "resolution_notes"}:
            value = str(value).strip() if value else None

        setattr(complaint, field_name, value)

    db.commit()
    db.refresh(complaint)

    return complaint

def get_dashboard_summary(
    db: Session,
    submitted_by_user_id: int | None = None,
) -> dict[str, int]:
    ownership_filter = None

    if submitted_by_user_id is not None:
        ownership_filter = Complaint.id.in_(
            select(ComplaintOwnership.complaint_id).where(
                ComplaintOwnership.submitted_by_user_id
                == submitted_by_user_id
            )
        )

    def count_with_filters(*conditions: Any) -> int:
        query = select(func.count()).select_from(Complaint)
        filters = list(conditions)

        if ownership_filter is not None:
            filters.append(ownership_filter)

        if filters:
            query = query.where(and_(*filters))

        return int(db.scalar(query) or 0)

    active_complaint_filter = Complaint.status.in_(["Open", "In Progress"])

    overdue_complaint_filter = Complaint.id.in_(
        select(ComplaintEscalation.complaint_id).where(
            ComplaintEscalation.due_at.is_not(None),
            ComplaintEscalation.due_at < datetime.utcnow(),
            ComplaintEscalation.escalation_state != "Escalated",
        )
    )

    escalated_complaint_filter = Complaint.id.in_(
        select(ComplaintEscalation.complaint_id).where(
            ComplaintEscalation.escalation_state == "Escalated"
        )
    )

    return {
        "total_complaints": count_with_filters(),
        "open_complaints": count_with_filters(Complaint.status == "Open"),
        "in_progress_complaints": count_with_filters(
            Complaint.status == "In Progress"
        ),
        "resolved_complaints": count_with_filters(
            Complaint.status == "Resolved"
        ),
        "closed_complaints": count_with_filters(
            Complaint.status == "Closed"
        ),
        "critical_open_complaints": count_with_filters(
            active_complaint_filter,
            Complaint.predicted_priority == "Critical",
        ),
        "high_open_complaints": count_with_filters(
            active_complaint_filter,
            Complaint.predicted_priority == "High",
        ),
        "possible_duplicate_complaints": count_with_filters(
            Complaint.possible_duplicate.is_(True)
        ),
        "overdue_open_complaints": count_with_filters(
            active_complaint_filter,
            overdue_complaint_filter,
        ),
        "escalated_open_complaints": count_with_filters(
            active_complaint_filter,
            escalated_complaint_filter,
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
    action: str | None = None,
    actor_user_id: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[AuditLog]]:
    filters = []

    if entity_type:
        filters.append(AuditLog.entity_type == entity_type)

    if entity_id:
        filters.append(AuditLog.entity_id == entity_id)

    if action:
        filters.append(AuditLog.action == action)

    if actor_user_id is not None:
        filters.append(AuditLog.actor_user_id == actor_user_id)

    query = select(AuditLog)
    count_query = select(func.count()).select_from(AuditLog)

    if filters:
        filter_expression = and_(*filters)
        query = query.where(filter_expression)
        count_query = count_query.where(filter_expression)

    total = int(db.scalar(count_query) or 0)

    logs = list(
        db.scalars(
            query.order_by(desc(AuditLog.created_at))
            .offset(offset)
            .limit(limit)
        ).all()
    )

    return total, logs


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


def list_unowned_complaints(
    db: Session,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[Complaint]]:
    ownership_exists = (
        select(ComplaintOwnership.id)
        .where(ComplaintOwnership.complaint_id == Complaint.id)
        .exists()
    )

    base_query = select(Complaint).where(~ownership_exists)

    total = int(
        db.scalar(
            select(func.count())
            .select_from(Complaint)
            .where(~ownership_exists)
        )
        or 0
    )

    complaints = list(
        db.scalars(
            base_query.order_by(desc(Complaint.created_at))
            .offset(offset)
            .limit(limit)
        ).all()
    )

    return total, complaints


def update_complaint_ownership(
    db: Session,
    ownership: ComplaintOwnership,
    submitted_by_user_id: int,
) -> ComplaintOwnership:
    ownership.submitted_by_user_id = submitted_by_user_id

    db.commit()
    db.refresh(ownership)

    return ownership


def get_complaint_assignment(
    db: Session,
    complaint_id: int,
) -> ComplaintAssignment | None:
    return db.scalar(
        select(ComplaintAssignment).where(
            ComplaintAssignment.complaint_id == complaint_id
        )
    )


def create_complaint_assignment(
    db: Session,
    complaint_id: int,
    assigned_to_user_id: int,
    assigned_department_id: int | None,
    assignment_note: str | None,
    assigned_by_user_id: int,
) -> ComplaintAssignment:
    assignment = ComplaintAssignment(
        complaint_id=complaint_id,
        assigned_to_user_id=assigned_to_user_id,
        assigned_department_id=assigned_department_id,
        assignment_note=assignment_note.strip() if assignment_note else None,
        assigned_by_user_id=assigned_by_user_id,
    )

    db.add(assignment)
    db.commit()
    db.refresh(assignment)

    return assignment


def update_complaint_assignment(
    db: Session,
    assignment: ComplaintAssignment,
    assigned_to_user_id: int,
    assigned_department_id: int | None,
    assignment_note: str | None,
    assigned_by_user_id: int,
) -> ComplaintAssignment:
    assignment.assigned_to_user_id = assigned_to_user_id
    assignment.assigned_department_id = assigned_department_id
    assignment.assignment_note = (
        assignment_note.strip() if assignment_note else None
    )
    assignment.assigned_by_user_id = assigned_by_user_id
    assignment.assigned_at = datetime.utcnow()

    db.commit()
    db.refresh(assignment)

    return assignment


def create_assignment_history(
    db: Session,
    complaint_id: int,
    previous_assigned_to_user_id: int | None,
    new_assigned_to_user_id: int,
    previous_department_id: int | None,
    new_department_id: int | None,
    changed_by_user_id: int,
    note: str | None = None,
) -> ComplaintAssignmentHistory:
    history = ComplaintAssignmentHistory(
        complaint_id=complaint_id,
        previous_assigned_to_user_id=previous_assigned_to_user_id,
        new_assigned_to_user_id=new_assigned_to_user_id,
        previous_department_id=previous_department_id,
        new_department_id=new_department_id,
        changed_by_user_id=changed_by_user_id,
        note=note.strip() if note else None,
    )

    db.add(history)
    db.commit()
    db.refresh(history)

    return history


def list_assignment_history(
    db: Session,
    complaint_id: int,
) -> list[ComplaintAssignmentHistory]:
    query = (
        select(ComplaintAssignmentHistory)
        .where(ComplaintAssignmentHistory.complaint_id == complaint_id)
        .order_by(ComplaintAssignmentHistory.changed_at.asc())
    )

    return list(db.scalars(query).all())


def get_complaint_escalation(
    db: Session,
    complaint_id: int,
) -> ComplaintEscalation | None:
    return db.scalar(
        select(ComplaintEscalation).where(
            ComplaintEscalation.complaint_id == complaint_id
        )
    )


def create_complaint_escalation(
    db: Session,
    complaint_id: int,
    due_at: datetime | None,
    escalation_state: str,
    escalation_reason: str | None,
    set_by_user_id: int,
) -> ComplaintEscalation:
    escalation = ComplaintEscalation(
        complaint_id=complaint_id,
        due_at=due_at,
        escalation_state=escalation_state,
        escalation_reason=(
            escalation_reason.strip()
            if escalation_reason
            else None
        ),
        set_by_user_id=set_by_user_id,
        escalated_at=(
            datetime.utcnow()
            if escalation_state == "Escalated"
            else None
        ),
    )

    db.add(escalation)
    db.commit()
    db.refresh(escalation)

    return escalation


def update_complaint_escalation(
    db: Session,
    escalation: ComplaintEscalation,
    due_at: datetime | None,
    escalation_state: str,
    escalation_reason: str | None,
    set_by_user_id: int,
) -> ComplaintEscalation:
    escalation.due_at = due_at
    escalation.escalation_state = escalation_state
    escalation.escalation_reason = (
        escalation_reason.strip()
        if escalation_reason
        else None
    )
    escalation.set_by_user_id = set_by_user_id
    escalation.escalated_at = (
        datetime.utcnow()
        if escalation_state == "Escalated"
        else None
    )

    db.commit()
    db.refresh(escalation)

    return escalation


def is_complaint_overdue(
    complaint: Complaint,
    escalation: ComplaintEscalation | None,
) -> bool:
    if escalation is None or escalation.due_at is None:
        return False

    if complaint.status in {"Resolved", "Closed"}:
        return False

    if escalation.escalation_state == "Escalated":
        return False

    return escalation.due_at < datetime.utcnow()


def can_staff_access_complaint(
    db: Session,
    complaint_id: int,
    staff_user_id: int,
    staff_department_id: int | None,
) -> bool:
    assignment = get_complaint_assignment(
        db=db,
        complaint_id=complaint_id,
    )

    # Unassigned complaints remain visible to staff.
    if assignment is None:
        return True

    # The explicitly assigned staff member always has access.
    if assignment.assigned_to_user_id == staff_user_id:
        return True

    # Other staff need membership in the assigned department.
    return (
        staff_department_id is not None
        and assignment.assigned_department_id == staff_department_id
    )


def list_training_eligible_records(
    db: Session,
) -> list[tuple[Complaint, MLFeedbackRecord, ComplaintVerification | None]]:
    query = (
        select(
            Complaint,
            MLFeedbackRecord,
            ComplaintVerification,
        )
        .join(
            MLFeedbackRecord,
            MLFeedbackRecord.complaint_id == Complaint.id,
        )
        .outerjoin(
            ComplaintVerification,
            ComplaintVerification.complaint_id == Complaint.id,
        )
        .where(
            MLFeedbackRecord.training_eligible.is_(True),
            Complaint.status.in_(["Resolved", "Closed"]),
        )
        .order_by(Complaint.id.asc())
    )

    return list(db.execute(query).all())


def get_ml_monitoring_summary(
    db: Session,
) -> dict[str, Any]:
    reviewed_feedback_records = int(
        db.scalar(
            select(func.count())
            .select_from(MLFeedbackRecord)
            .where(MLFeedbackRecord.reviewed_at.is_not(None))
        )
        or 0
    )

    training_eligible_records = int(
        db.scalar(
            select(func.count())
            .select_from(MLFeedbackRecord)
            .where(MLFeedbackRecord.training_eligible.is_(True))
        )
        or 0
    )

    verified_impact_records = int(
        db.scalar(
            select(func.count())
            .select_from(ComplaintVerification)
            .where(
                ComplaintVerification.impact_verification_status.in_(
                    ["Verified", "Adjusted"]
                ),
                ComplaintVerification.verified_affected_population.is_not(
                    None
                ),
            )
        )
        or 0
    )

    feedback_rows = list(
        db.execute(
            select(Complaint, MLFeedbackRecord)
            .join(
                MLFeedbackRecord,
                MLFeedbackRecord.complaint_id == Complaint.id,
            )
            .where(MLFeedbackRecord.reviewed_at.is_not(None))
        ).all()
    )

    def agreement_metric(
        predicted_getter: Any,
        final_getter: Any,
    ) -> dict[str, Any]:
        evaluated_rows = [
            (complaint, feedback)
            for complaint, feedback in feedback_rows
            if predicted_getter(complaint) is not None
            and final_getter(feedback) is not None
        ]

        matching_records = sum(
            1
            for complaint, feedback in evaluated_rows
            if predicted_getter(complaint) == final_getter(feedback)
        )

        evaluated_records = len(evaluated_rows)

        return {
            "evaluated_records": evaluated_records,
            "matching_records": matching_records,
            "agreement_rate": (
                matching_records / evaluated_records
                if evaluated_records > 0
                else None
            ),
        }

    category_agreement = agreement_metric(
        predicted_getter=lambda complaint: complaint.predicted_category,
        final_getter=lambda feedback: feedback.final_category,
    )

    priority_agreement = agreement_metric(
        predicted_getter=lambda complaint: complaint.predicted_priority,
        final_getter=lambda feedback: feedback.final_priority,
    )

    department_rows = [
        (complaint, feedback)
        for complaint, feedback in feedback_rows
        if complaint.assigned_department is not None
        and feedback.final_department_id is not None
    ]

    department_matching_records = 0

    for complaint, feedback in department_rows:
        department = get_department_by_id(
            db=db,
            department_id=feedback.final_department_id,
        )

        if department is not None:
            predicted_department = complaint.assigned_department.strip().lower()
            final_department_name = department.name.strip().lower()

            if predicted_department == final_department_name:
                department_matching_records += 1

    department_evaluated_records = len(department_rows)

    department_agreement = {
        "evaluated_records": department_evaluated_records,
        "matching_records": department_matching_records,
        "agreement_rate": (
            department_matching_records / department_evaluated_records
            if department_evaluated_records > 0
            else None
        ),
    }

    resolution_rows = [
        (complaint, feedback)
        for complaint, feedback in feedback_rows
        if complaint.estimated_resolution_hours is not None
        and feedback.actual_resolution_hours is not None
    ]

    absolute_errors = [
        abs(
            complaint.estimated_resolution_hours
            - feedback.actual_resolution_hours
        )
        for complaint, feedback in resolution_rows
    ]

    resolution_time = {
        "evaluated_records": len(absolute_errors),
        "mean_absolute_error_hours": (
            sum(absolute_errors) / len(absolute_errors)
            if absolute_errors
            else None
        ),
    }

    duplicate_rows = [
        (complaint, feedback)
        for complaint, feedback in feedback_rows
        if feedback.duplicate_decision is not None
    ]

    suggested_duplicate_records = sum(
        1
        for complaint, _feedback in duplicate_rows
        if complaint.possible_duplicate
    )

    confirmed_duplicate_records = sum(
        1
        for _complaint, feedback in duplicate_rows
        if feedback.duplicate_decision == "ConfirmedDuplicate"
    )

    rejected_duplicate_suggestions = sum(
        1
        for complaint, feedback in duplicate_rows
        if complaint.possible_duplicate
        and feedback.duplicate_decision == "NotDuplicate"
    )

    related_issue_records = sum(
        1
        for _complaint, feedback in duplicate_rows
        if feedback.duplicate_decision == "RelatedIssue"
    )

    duplicate_decisions = {
        "evaluated_records": len(duplicate_rows),
        "suggested_duplicate_records": suggested_duplicate_records,
        "confirmed_duplicate_records": confirmed_duplicate_records,
        "rejected_duplicate_suggestions": rejected_duplicate_suggestions,
        "related_issue_records": related_issue_records,
    }

    return {
        "reviewed_feedback_records": reviewed_feedback_records,
        "training_eligible_records": training_eligible_records,
        "verified_impact_records": verified_impact_records,
        "category_agreement": category_agreement,
        "department_agreement": department_agreement,
        "priority_agreement": priority_agreement,
        "resolution_time": resolution_time,
        "duplicate_decisions": duplicate_decisions,
    }