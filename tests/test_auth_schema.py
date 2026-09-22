from apps.api.schemas.auth import LoginRequest


def test_login_request_normalizes_email() -> None:
    request = LoginRequest(
        email="  MAINTENANCE.STAFF@CAMPUSRESOLVE.EXAMPLE.COM  ",
        password="test-password",
    )

    assert str(request.email) == "maintenance.staff@campusresolve.example.com"