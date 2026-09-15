from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AuditLogResponse(BaseModel):
    id: int
    actor_user_id: int | None = None
    action: str
    entity_type: str
    entity_id: str
    details: str | None = None
    created_at: datetime


class AuditLogListResponse(BaseModel):
    total: int
    logs: list[AuditLogResponse]