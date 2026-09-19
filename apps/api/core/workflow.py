from __future__ import annotations

from collections.abc import Mapping

ComplaintStatus = str
ImpactVerificationStatus = str
FinalPriority = str
DuplicateDecision = str
EscalationState = str


COMPLAINT_STATUSES: frozenset[ComplaintStatus] = frozenset(
    {
        "Open",
        "In Progress",
        "Resolved",
        "Closed",
    }
)


ALLOWED_STATUS_TRANSITIONS: Mapping[
    ComplaintStatus,
    frozenset[ComplaintStatus],
] = {
    "Open": frozenset({"In Progress", "Closed"}),
    "In Progress": frozenset({"Resolved", "Closed"}),
    "Resolved": frozenset({"In Progress", "Closed"}),
    "Closed": frozenset(),
}


IMPACT_VERIFICATION_STATUSES: frozenset[
    ImpactVerificationStatus
] = frozenset(
    {
        "Unverified",
        "Verified",
        "Adjusted",
        "Rejected",
    }
)


FINAL_PRIORITIES: frozenset[FinalPriority] = frozenset(
    {
        "Low",
        "Medium",
        "High",
        "Critical",
    }
)


DUPLICATE_DECISIONS: frozenset[DuplicateDecision] = frozenset(
    {
        "NotReviewed",
        "ConfirmedDuplicate",
        "NotDuplicate",
        "RelatedIssue",
    }
)


ESCALATION_STATES: frozenset[EscalationState] = frozenset(
    {
        "OnTrack",
        "Escalated",
    }
)


def is_valid_status(status: str) -> bool:
    return status in COMPLAINT_STATUSES


def can_transition_status(
    current_status: str,
    new_status: str,
) -> bool:
    return new_status in ALLOWED_STATUS_TRANSITIONS.get(
        current_status,
        frozenset(),
    )


def is_valid_impact_verification_status(status: str) -> bool:
    return status in IMPACT_VERIFICATION_STATUSES


def is_valid_final_priority(priority: str) -> bool:
    return priority in FINAL_PRIORITIES


def is_valid_duplicate_decision(decision: str) -> bool:
    return decision in DUPLICATE_DECISIONS


def is_valid_escalation_state(state: str) -> bool:
    return state in ESCALATION_STATES