from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ComplaintStatusHistoryResponse(BaseModel):
    id: int
    old_status: str | None = None
    new_status: str
    note: str | None = None
    changed_by_user_id: int | None = None
    changed_at: datetime


class ComplaintStatusHistoryListResponse(BaseModel):
    complaint_reference: str
    total: int
    history: list[ComplaintStatusHistoryResponse]