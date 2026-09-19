from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ComplaintAssignmentHistoryEventResponse(BaseModel):
    previous_assigned_to_user_id: int | None = None
    new_assigned_to_user_id: int
    previous_department_id: int | None = None
    new_department_id: int | None = None
    note: str | None = None
    changed_at: datetime


class ComplaintAssignmentHistoryListResponse(BaseModel):
    complaint_reference: str
    total: int
    history: list[ComplaintAssignmentHistoryEventResponse]