from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class TimelineStatusEventResponse(BaseModel):
    old_status: str | None = None
    new_status: str
    note: str | None = None
    changed_at: datetime


class TimelineAssignmentResponse(BaseModel):
    assigned_to_user_id: int
    assigned_department_id: int | None = None
    assignment_note: str | None = None
    assigned_at: datetime


class TimelineImpactResponse(BaseModel):
    reported_affected_population: int
    verified_affected_population: int | None = None
    impact_verification_status: str
    impact_verification_note: str | None = None
    verified_at: datetime | None = None

class TimelineEscalationResponse(BaseModel):
    due_at: datetime | None = None
    escalation_state: str
    is_overdue: bool
    escalated_at: datetime | None = None
    escalation_reason: str | None = None


class ComplaintTimelineResponse(BaseModel):
    complaint_reference: str
    complaint_text: str
    location_type: str
    specific_location: str
    status: str
    created_at: datetime
    updated_at: datetime
    assignment: TimelineAssignmentResponse | None = None
    impact: TimelineImpactResponse | None = None
    escalation: TimelineEscalationResponse | None = None
    status_history: list[TimelineStatusEventResponse]
    