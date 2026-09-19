from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
)


ImpactVerificationStatus = Literal[
    "Unverified",
    "Verified",
    "Adjusted",
    "Disputed",
]


class DepartmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(ge=1)
    code: str
    name: str
    description: str | None = None
    contact_email: EmailStr | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class LocationTypeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(ge=1)
    name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class CampusBlockResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(ge=1)
    code: str
    name: str
    location_type_id: int = Field(ge=1)
    capacity: int | None = Field(default=None, ge=0)
    responsible_department_id: int | None = Field(
        default=None,
        ge=1,
    )
    is_active: bool
    created_at: datetime
    updated_at: datetime


class DepartmentListResponse(BaseModel):
    total: int = Field(ge=0)
    departments: list[DepartmentResponse] = Field(
        default_factory=list
    )


class LocationTypeListResponse(BaseModel):
    total: int = Field(ge=0)
    location_types: list[LocationTypeResponse] = Field(
        default_factory=list
    )


class CampusBlockListResponse(BaseModel):
    total: int = Field(ge=0)
    campus_blocks: list[CampusBlockResponse] = Field(
        default_factory=list
    )


class DepartmentCreateRequest(BaseModel):
    code: str = Field(
        min_length=2,
        max_length=30,
        pattern=r"^[A-Za-z0-9_-]+$",
        examples=["MAINT"],
    )
    name: str = Field(
        min_length=2,
        max_length=150,
        examples=["Maintenance Department"],
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
    )
    contact_email: EmailStr | None = None

    @field_validator("code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Department name cannot be blank.")

        return value

    @field_validator("description")
    @classmethod
    def normalize_description(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class CampusBlockCreateRequest(BaseModel):
    code: str = Field(
        min_length=2,
        max_length=40,
        pattern=r"^[A-Za-z0-9_-]+$",
        examples=["HB-D"],
    )
    name: str = Field(
        min_length=2,
        max_length=150,
        examples=["Hostel Block D"],
    )
    location_type_id: int = Field(
        ge=1,
        examples=[1],
    )
    capacity: int | None = Field(
        default=None,
        ge=0,
        le=100000,
        examples=[240],
    )
    responsible_department_id: int | None = Field(
        default=None,
        ge=1,
        examples=[3],
    )

    @field_validator("code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Campus-block name cannot be blank.")

        return value


class DepartmentUpdateRequest(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
    )
    contact_email: EmailStr | None = None
    is_active: bool | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None

    @field_validator("description")
    @classmethod
    def normalize_description(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class CampusBlockUpdateRequest(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )
    location_type_id: int | None = Field(default=None, ge=1)
    capacity: int | None = Field(
        default=None,
        ge=0,
        le=100000,
    )
    responsible_department_id: int | None = Field(
        default=None,
        ge=1,
    )
    is_active: bool | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class ComplaintImpactVerificationRequest(BaseModel):
    verified_affected_population: int | None = Field(
        default=None,
        ge=0,
        le=100000,
        examples=[120],
    )

    impact_verification_status: ImpactVerificationStatus = Field(
        examples=["Verified"],
    )

    impact_verification_note: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator("impact_verification_note")
    @classmethod
    def normalize_verification_note(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None