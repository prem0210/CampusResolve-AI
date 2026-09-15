from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


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


class ComplaintAssignmentResponse(BaseModel):
    complaint_reference: str
    assigned_to_user_id: int
    assigned_department_id: int | None = None
    assignment_note: str | None = None
    assigned_by_user_id: int
    assigned_at: datetime
    updated_at: datetime