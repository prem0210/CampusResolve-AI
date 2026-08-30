from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from datetime import datetime

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
    safety_flag: int = Field(
        default=0,
        ge=0,
        le=1,
    )
    repeat_count: int = Field(
        default=0,
        ge=0,
        le=100,
    )


class FeatureContribution(BaseModel):
    feature: str
    contribution: float | None = None
    shap_value: float | None = None
    direction: str | None = None


class CategoryPredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_features: list[FeatureContribution]


class PriorityPredictionResponse(BaseModel):
    predicted_priority: str
    confidence: float
    top_features: list[FeatureContribution]


class DuplicateCandidate(BaseModel):
    complaint_id: str
    complaint_text: str
    category: str
    priority: str
    similarity_score: float


class ComplaintPredictionResponse(BaseModel):
    predicted_category: str
    category_confidence: float
    assigned_department: str
    predicted_priority: str
    priority_confidence: float
    estimated_resolution_hours: float
    prediction_interval_plus_minus_hours: float
    duplicate_threshold: float
    possible_duplicate: bool
    duplicate_candidates: list[DuplicateCandidate]
    category_explanation: CategoryPredictionResponse
    priority_explanation: PriorityPredictionResponse
    explanation: str


class ComplaintCreateResponse(ComplaintPredictionResponse):
    complaint_reference: str
    status: str
    created_at: datetime



ComplaintStatus = Literal["Open", "In Progress", "Resolved", "Closed"]


class StoredComplaintResponse(BaseModel):
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


class ComplaintQueueResponse(BaseModel):
    total: int
    complaints: list[StoredComplaintResponse]


class DashboardSummaryResponse(BaseModel):
    total_complaints: int
    open_complaints: int
    in_progress_complaints: int
    resolved_complaints: int
    closed_complaints: int
    critical_open_complaints: int
    high_open_complaints: int
    possible_duplicate_complaints: int