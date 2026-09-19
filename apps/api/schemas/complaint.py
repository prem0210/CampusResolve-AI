from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ComplaintStatus = Literal["Open", "In Progress", "Resolved", "Closed"]
ComplaintAssignmentState = Literal["Assigned", "Unassigned"]
ComplaintEscalationState = Literal["OnTrack", "Overdue", "Escalated"]


class ORMResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ComplaintPredictionRequest(BaseModel):
    complaint_text: str = Field(
        min_length=5,
        max_length=2000,
        description="Campus complaint submitted by a user.",
    )
    language: Literal["en", "ta", "ta_en"] = "en"
    location_type: str = Field(
        default="Campus Building",
        min_length=2,
        max_length=100,
    )
    specific_location: str = Field(
        default="Not specified",
        min_length=2,
        max_length=150,
    )
    affected_population: int = Field(
        default=1,
        ge=1,
        le=10000,
    )
    safety_flag: bool = False
    repeat_count: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    @field_validator(
        "complaint_text",
        "location_type",
        "specific_location",
    )
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("This field cannot be blank.")

        return value


class FeatureContribution(BaseModel):
    feature: str
    contribution: float | None = None
    shap_value: float | None = None
    direction: str | None = None


class CategoryPredictionResponse(BaseModel):
    predicted_category: str
    confidence: float = Field(ge=0, le=1)
    top_features: list[FeatureContribution] = Field(default_factory=list)


class PriorityPredictionResponse(BaseModel):
    predicted_priority: str
    confidence: float = Field(ge=0, le=1)
    top_features: list[FeatureContribution] = Field(default_factory=list)


class DuplicateCandidate(BaseModel):
    complaint_id: str
    complaint_text: str
    category: str
    priority: str
    similarity_score: float = Field(ge=0, le=1)


class ComplaintPredictionResponse(BaseModel):
    predicted_category: str
    category_confidence: float = Field(ge=0, le=1)
    assigned_department: str
    predicted_priority: str
    priority_confidence: float = Field(ge=0, le=1)
    estimated_resolution_hours: float = Field(ge=0)
    prediction_interval_plus_minus_hours: float = Field(ge=0)
    duplicate_threshold: float = Field(ge=0, le=1)
    possible_duplicate: bool
    duplicate_candidates: list[DuplicateCandidate] = Field(
        default_factory=list
    )
    category_explanation: CategoryPredictionResponse
    priority_explanation: PriorityPredictionResponse
    explanation: str


class ComplaintCreateResponse(ComplaintPredictionResponse):
    complaint_reference: str
    status: ComplaintStatus
    created_at: datetime


class StoredComplaintResponse(ORMResponseModel):
    complaint_reference: str
    complaint_text: str
    language: str
    location_type: str
    specific_location: str
    affected_population: int
    safety_flag: bool
    repeat_count: int

    predicted_category: str
    category_confidence: float
    assigned_department: str

    predicted_priority: str
    priority_confidence: float

    estimated_resolution_hours: float
    prediction_interval_plus_minus_hours: float

    duplicate_threshold: float | None = None
    possible_duplicate: bool
    top_duplicate_id: str | None = None
    top_duplicate_similarity: float | None = None

    status: ComplaintStatus
    staff_notes: str | None = None

    created_at: datetime
    updated_at: datetime


class ComplaintStatusUpdateRequest(BaseModel):
    status: ComplaintStatus | None = None
    staff_notes: str | None = Field(
        default=None,
        max_length=3000,
    )

    @field_validator("staff_notes")
    @classmethod
    def normalize_staff_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class ComplaintQueueResponse(BaseModel):
    total: int = Field(ge=0)
    complaints: list[StoredComplaintResponse] = Field(
        default_factory=list
    )


class DashboardSummaryResponse(BaseModel):
    total_complaints: int = Field(ge=0)
    open_complaints: int = Field(ge=0)
    in_progress_complaints: int = Field(ge=0)
    resolved_complaints: int = Field(ge=0)
    closed_complaints: int = Field(ge=0)
    critical_open_complaints: int = Field(ge=0)
    high_open_complaints: int = Field(ge=0)
    possible_duplicate_complaints: int = Field(ge=0)
    overdue_open_complaints: int = Field(ge=0)
    escalated_open_complaints: int = Field(ge=0)


class ComplaintListFilters(BaseModel):
    status: ComplaintStatus | None = None
    priority: str | None = Field(
        default=None,
        max_length=100,
    )
    department: str | None = Field(
        default=None,
        max_length=150,
    )
    category: str | None = Field(
        default=None,
        max_length=150,
    )
    assigned_to_user_id: int | None = Field(
        default=None,
        ge=1,
    )
    assigned_department_id: int | None = Field(
        default=None,
        ge=1,
    )
    assignment_state: ComplaintAssignmentState | None = None
    escalation_state: ComplaintEscalationState | None = None
    due_before: datetime | None = None
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)