from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComplaintAssignmentRequest(BaseModel):
    assigned_to_user_id: int = Field(
        ge=1,
        examples=[2],
    )

    assigned_department_id: int | None = Field(
        default=None,
        ge=1,
        examples=[1],
    )

    assignment_note: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator("assignment_note")
    @classmethod
    def normalize_assignment_note(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class ComplaintAssignmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    complaint_reference: str
    assigned_to_user_id: int = Field(ge=1)
    assigned_department_id: int | None = Field(default=None, ge=1)
    assignment_note: str | None = None
    assigned_by_user_id: int = Field(ge=1)
    assigned_at: datetime
    updated_at: datetime