from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ComplaintOwnershipUpdateRequest(BaseModel):
    submitted_by_user_id: int = Field(
        ge=1,
        examples=[1],
    )


class ComplaintOwnershipResponse(BaseModel):
    complaint_reference: str
    submitted_by_user_id: int
    created_at: datetime


class UnownedComplaintResponse(BaseModel):
    complaint_reference: str
    complaint_text: str
    status: str
    created_at: datetime


class UnownedComplaintListResponse(BaseModel):
    total: int
    complaints: list[UnownedComplaintResponse]