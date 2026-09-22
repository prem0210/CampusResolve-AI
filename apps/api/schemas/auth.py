from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=128,
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).strip().lower()


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(ge=1)
    full_name: str
    email: EmailStr
    role: str
    department_id: int | None = Field(default=None, ge=1)
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str = Field(min_length=1)
    token_type: str = "bearer"
    expires_in_minutes: int = Field(gt=0)
    user: UserResponse
