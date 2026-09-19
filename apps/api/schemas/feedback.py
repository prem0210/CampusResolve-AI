from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


FinalPriority = Literal[
    "Low",
    "Medium",
    "High",
    "Critical",
]

DuplicateDecision = Literal[
    "NotReviewed",
    "ConfirmedDuplicate",
    "NotDuplicate",
    "RelatedIssue",
]


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

    final_priority: FinalPriority | None = Field(
        default=None,
        examples=["High"],
    )

    actual_resolution_hours: float | None = Field(
        default=None,
        ge=0,
        le=8760,
        examples=[18.5],
    )

    duplicate_decision: DuplicateDecision | None = Field(
        default=None,
        examples=["NotDuplicate"],
    )

    training_eligible: bool | None = None

    exclusion_reason: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator(
        "final_category",
        "exclusion_reason",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class MLFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    complaint_reference: str
    final_category: str | None = None
    final_department_id: int | None = Field(
        default=None,
        ge=1,
    )
    final_priority: FinalPriority | None = None
    actual_resolution_hours: float | None = Field(
        default=None,
        ge=0,
    )
    duplicate_decision: DuplicateDecision | None = None
    training_eligible: bool
    exclusion_reason: str | None = None
    reviewed_by_user_id: int | None = Field(
        default=None,
        ge=1,
    )
    reviewed_at: datetime | None = None