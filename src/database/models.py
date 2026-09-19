from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    __table_args__ = (
        CheckConstraint(
            "role IN ('Student', 'Staff', 'Admin')",
            name="ck_users_role",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    full_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        default="Student",
        nullable=False,
        index=True,
    )

    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"),
        nullable=True,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    code: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        index=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        unique=True,
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    contact_email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class LocationType(Base):
    __tablename__ = "location_types"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class CampusBlock(Base):
    __tablename__ = "campus_blocks"

    __table_args__ = (
        CheckConstraint(
            "capacity IS NULL OR capacity >= 0",
            name="ck_campus_blocks_capacity_nonnegative",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    code: Mapped[str] = mapped_column(
        String(40),
        unique=True,
        index=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        unique=True,
        nullable=False,
    )

    location_type_id: Mapped[int] = mapped_column(
        ForeignKey("location_types.id"),
        nullable=False,
        index=True,
    )

    capacity: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    responsible_department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"),
        nullable=True,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    entity_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    entity_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    details: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )


class Complaint(Base):
    __tablename__ = "complaints"

    __table_args__ = (
        CheckConstraint(
            "affected_population >= 1",
            name="ck_complaints_affected_population_positive",
        ),
        CheckConstraint(
            "repeat_count >= 0",
            name="ck_complaints_repeat_count_nonnegative",
        ),
        CheckConstraint(
            "category_confidence >= 0 AND category_confidence <= 1",
            name="ck_complaints_category_confidence_range",
        ),
        CheckConstraint(
            "priority_confidence >= 0 AND priority_confidence <= 1",
            name="ck_complaints_priority_confidence_range",
        ),
        CheckConstraint(
            "estimated_resolution_hours >= 0",
            name="ck_complaints_resolution_hours_nonnegative",
        ),
        CheckConstraint(
            "prediction_interval_plus_minus_hours >= 0",
            name="ck_complaints_resolution_interval_nonnegative",
        ),
        CheckConstraint(
            "duplicate_threshold >= 0 AND duplicate_threshold <= 1",
            name="ck_complaints_duplicate_threshold_range",
        ),
        CheckConstraint(
            "top_duplicate_similarity IS NULL OR "
            "(top_duplicate_similarity >= 0 AND top_duplicate_similarity <= 1)",
            name="ck_complaints_duplicate_similarity_range",
        ),
        CheckConstraint(
            "status IN ('Open', 'In Progress', 'Resolved', 'Closed')",
            name="ck_complaints_status",
        ),
        CheckConstraint(
            "predicted_priority IN ('Low', 'Medium', 'High', 'Critical')",
            name="ck_complaints_predicted_priority",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_reference: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        index=True,
        nullable=False,
    )

    complaint_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    language: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    location_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    specific_location: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    affected_population: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    safety_flag: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    repeat_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    predicted_category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    category_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    assigned_department: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    predicted_priority: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    priority_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    estimated_resolution_hours: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    prediction_interval_plus_minus_hours: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    duplicate_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    possible_duplicate: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    # Keep String(30) because your prediction schema exposes complaint_id as str.
    # Change to Integer + ForeignKey("complaints.id") only if it truly stores a
    # numeric database primary key.
    top_duplicate_id: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    top_duplicate_similarity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="Open",
        nullable=False,
        index=True,
    )

    staff_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class ComplaintVerification(Base):
    __tablename__ = "complaint_verifications"

    __table_args__ = (
        CheckConstraint(
            "reported_affected_population >= 1",
            name="ck_complaint_verifications_reported_population_positive",
        ),
        CheckConstraint(
            "verified_affected_population IS NULL OR "
            "verified_affected_population >= 0",
            name="ck_complaint_verifications_verified_population_nonnegative",
        ),
        CheckConstraint(
            "impact_verification_status IN "
            "('Unverified', 'Verified', 'Adjusted', 'Rejected')",
            name="ck_complaint_verifications_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    reported_affected_population: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    verified_affected_population: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    impact_verification_status: Mapped[str] = mapped_column(
        String(30),
        default="Unverified",
        nullable=False,
        index=True,
    )

    impact_verification_note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    verified_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class ComplaintStatusHistory(Base):
    __tablename__ = "complaint_status_history"

    __table_args__ = (
        CheckConstraint(
            "old_status IS NULL OR "
            "old_status IN ('Open', 'In Progress', 'Resolved', 'Closed')",
            name="ck_complaint_status_history_old_status",
        ),
        CheckConstraint(
            "new_status IN ('Open', 'In Progress', 'Resolved', 'Closed')",
            name="ck_complaint_status_history_new_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id"),
        nullable=False,
        index=True,
    )

    old_status: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    new_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    changed_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )


class MLFeedbackRecord(Base):
    __tablename__ = "ml_feedback_records"

    __table_args__ = (
        CheckConstraint(
            "actual_resolution_hours IS NULL OR actual_resolution_hours >= 0",
            name="ck_ml_feedback_actual_resolution_nonnegative",
        ),
        CheckConstraint(
            "final_priority IS NULL OR "
            "final_priority IN ('Low', 'Medium', 'High', 'Critical')",
            name="ck_ml_feedback_final_priority",
        ),
        CheckConstraint(
            "duplicate_decision IS NULL OR duplicate_decision IN "
            "('NotReviewed', 'ConfirmedDuplicate', 'NotDuplicate', 'RelatedIssue')",
            name="ck_ml_feedback_duplicate_decision",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    final_category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    final_department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"),
        nullable=True,
        index=True,
    )

    final_priority: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    actual_resolution_hours: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    duplicate_decision: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    training_eligible: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    exclusion_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reviewed_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class ComplaintOwnership(Base):
    __tablename__ = "complaint_ownership"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    submitted_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )


class ComplaintAssignment(Base):
    __tablename__ = "complaint_assignments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    assigned_to_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    assigned_department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"),
        nullable=True,
        index=True,
    )

    assignment_note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    assigned_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class ComplaintAssignmentHistory(Base):
    __tablename__ = "complaint_assignment_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id"),
        nullable=False,
        index=True,
    )

    previous_assigned_to_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    new_assigned_to_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    previous_department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"),
        nullable=True,
        index=True,
    )

    new_department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"),
        nullable=True,
        index=True,
    )

    note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    changed_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )


class ComplaintEscalation(Base):
    __tablename__ = "complaint_escalations"

    __table_args__ = (
        CheckConstraint(
            "escalation_state IN ('OnTrack', 'Escalated')",
            name="ck_complaint_escalations_state",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    escalation_state: Mapped[str] = mapped_column(
        String(30),
        default="OnTrack",
        nullable=False,
        index=True,
    )

    escalated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    escalation_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    set_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )