import pytest
from pydantic import ValidationError

from apps.api.schemas.complaint import ComplaintPredictionRequest

from apps.api.schemas.master_data import (
    ComplaintImpactVerificationRequest,
)

from apps.api.core.workflow import can_transition_status

VALID_PAYLOAD = {
    "complaint_text": "Water is leaking from the pipe near Hostel Block B.",
    "language": "en",
    "location_type": "Hostel",
    "specific_location": "Hostel Block B",
    "affected_population": 40,
    "safety_flag": 0,
    "repeat_count": 1,
}


def test_valid_complaint_request() -> None:
    request = ComplaintPredictionRequest(**VALID_PAYLOAD)

    assert request.complaint_text == VALID_PAYLOAD["complaint_text"]
    assert request.language == "en"
    assert request.affected_population == 40


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("complaint_text", "bad"),
        ("language", "hindi"),
        ("affected_population", 0),
        ("safety_flag", 2),
        ("repeat_count", -1),
    ],
)
def test_invalid_complaint_request_is_rejected(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = VALID_PAYLOAD.copy()
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        ComplaintPredictionRequest(**payload)

def test_rejected_impact_verification_status_is_accepted() -> None:
    request = ComplaintImpactVerificationRequest(
        impact_verification_status="Rejected",
        impact_verification_note="The reported impact could not be verified.",
    )

    assert request.impact_verification_status == "Rejected"
    assert (
        request.impact_verification_note
        == "The reported impact could not be verified."
    )


def test_disputed_impact_verification_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ComplaintImpactVerificationRequest(
            impact_verification_status="Disputed",
        )

@pytest.mark.parametrize(
    ("current_status", "new_status"),
    [
        ("Open", "In Progress"),
        ("Open", "Closed"),
        ("In Progress", "Resolved"),
        ("In Progress", "Closed"),
        ("Resolved", "In Progress"),
        ("Resolved", "Closed"),
    ],
)
def test_allowed_status_transitions(
    current_status: str,
    new_status: str,
) -> None:
    assert can_transition_status(current_status, new_status) is True


@pytest.mark.parametrize(
    ("current_status", "new_status"),
    [
        ("Open", "Resolved"),
        ("In Progress", "Open"),
        ("Resolved", "Open"),
        ("Closed", "Open"),
        ("Closed", "In Progress"),
        ("Closed", "Resolved"),
        ("Closed", "Closed"),
    ],
)
def test_disallowed_status_transitions(
    current_status: str,
    new_status: str,
) -> None:
    assert can_transition_status(current_status, new_status) is False