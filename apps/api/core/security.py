from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jose import JWTError, jwt
from pwdlib import PasswordHash


PASSWORD_HASHER = PasswordHash.recommended()

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "campusresolve-api").strip()
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "campusresolve-client").strip()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()

if not JWT_SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY must be configured in the environment."
    )

if JWT_ALGORITHM != "HS256":
    raise RuntimeError(
        "This security module is configured for HS256. "
        "Use a dedicated asymmetric-key implementation before selecting "
        "another algorithm."
    )

try:
    ACCESS_TOKEN_EXPIRE_MINUTES = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
except ValueError as error:
    raise RuntimeError(
        "ACCESS_TOKEN_EXPIRE_MINUTES must be an integer."
    ) from error

if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
    raise RuntimeError(
        "ACCESS_TOKEN_EXPIRE_MINUTES must be greater than zero."
    )


def hash_password(password: str) -> str:
    password = password.strip()

    if not password:
        raise ValueError("Password cannot be blank.")

    return PASSWORD_HASHER.hash(password)


def verify_password(
    plain_password: str,
    password_hash: str,
) -> bool:
    if not plain_password or not password_hash:
        return False

    return PASSWORD_HASHER.verify(
        plain_password,
        password_hash,
    )


def create_access_token(
    subject: str,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    expiry = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role.strip(),
        "type": "access",
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "exp": expiry,
        "jti": str(uuid4()),
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
        )
    except JWTError:
        return None

    subject = payload.get("sub")
    token_type = payload.get("type")

    if not isinstance(subject, str) or not subject.strip():
        return None

    if token_type != "access":
        return None

    return payload