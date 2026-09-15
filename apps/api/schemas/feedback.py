from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MLFeedbackUpdateRequest(BaseModel):
    final_category: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        examples=["Water Leakage"],
    )

    final_department_id: int | None = Field(
        default=None,
        ge=1,
        examples=[1],
    )

    final_priority: str | None = Field(
        default=None,
        examples=["High"],
    )

    actual_resolution_hours: float | None = Field(
        default=None,
        ge=0,
        le=8760,
        examples=[18.5],
    )

    duplicate_decision: str | None = Field(
        default=None,
        examples=["NotDuplicate"],
    )

    training_eligible: bool | None = None

    exclusion_reason: str | None = Field(
        default=None,
        max_length=1000,
    )


class MLFeedbackResponse(BaseModel):
    complaint_reference: str
    final_category: str | None = None
    final_department_id: int | None = None
    final_priority: str | None = None
    actual_resolution_hours: float | None = None
    duplicate_decision: str | None = None
    training_eligible: bool
    exclusion_reason: str | None = None
    reviewed_by_user_id: int | None = None
    reviewed_at: datetime | None = None