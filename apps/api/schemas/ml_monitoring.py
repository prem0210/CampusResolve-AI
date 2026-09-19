from __future__ import annotations

from pydantic import BaseModel


class AgreementMetricResponse(BaseModel):
    evaluated_records: int
    matching_records: int
    agreement_rate: float | None = None


class ResolutionTimeMetricResponse(BaseModel):
    evaluated_records: int
    mean_absolute_error_hours: float | None = None


class DuplicateDecisionMetricResponse(BaseModel):
    evaluated_records: int
    suggested_duplicate_records: int
    confirmed_duplicate_records: int
    rejected_duplicate_suggestions: int
    related_issue_records: int


class MLMonitoringResponse(BaseModel):
    reviewed_feedback_records: int
    training_eligible_records: int
    verified_impact_records: int
    category_agreement: AgreementMetricResponse
    department_agreement: AgreementMetricResponse
    priority_agreement: AgreementMetricResponse
    resolution_time: ResolutionTimeMetricResponse
    duplicate_decisions: DuplicateDecisionMetricResponse