from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from apps.api.core.security import decode_access_token
from src.database.database import get_db
from src.database.models import User
from src.database.repository import get_user_by_id


bearer_scheme = HTTPBearer(auto_error=False)


def authentication_exception(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise authentication_exception("Authentication is required.")

    token_data = decode_access_token(credentials.credentials)

    if token_data is None:
        raise authentication_exception("Invalid or expired access token.")

    subject = token_data.get("sub")

    if subject is None:
        raise authentication_exception("Invalid access token subject.")

    try:
        user_id = int(subject)
    except (TypeError, ValueError) as error:
        raise authentication_exception(
            "Invalid access token subject."
        ) from error

    user = get_user_by_id(db=db, user_id=user_id)

    if user is None or not user.is_active:
        raise authentication_exception("User account is unavailable.")

    return user


def require_roles(*allowed_roles: str) -> Callable[..., User]:
    normalized_roles = {
        role.strip().lower()
        for role in allowed_roles
        if role.strip()
    }

    if not normalized_roles:
        raise ValueError("At least one allowed role is required.")

    def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        current_role = current_user.role.strip().lower()

        if current_role not in normalized_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )

        return current_user

    return role_checker


require_student = require_roles("Student")
require_staff = require_roles("Staff")
require_admin = require_roles("Admin")
require_staff_or_admin = require_roles("Staff", "Admin")