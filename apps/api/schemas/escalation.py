from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EscalationState = Literal["OnTrack", "Escalated"]


class ComplaintEscalationUpdateRequest(BaseModel):
    due_at: datetime | None = None

    escalation_state: EscalationState | None = Field(
        default=None,
        examples=["Escalated"],
    )

    escalation_reason: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator("escalation_reason")
    @classmethod
    def normalize_escalation_reason(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class ComplaintEscalationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    complaint_reference: str
    due_at: datetime | None = None
    escalation_state: EscalationState
    is_overdue: bool
    escalated_at: datetime | None = None
    escalation_reason: str | None = None
    set_by_user_id: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime