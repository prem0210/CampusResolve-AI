import pytest
from pydantic import ValidationError

from apps.api.schemas.complaint import ComplaintPredictionRequest


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