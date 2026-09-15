from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DepartmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str
    name: str
    description: str | None = None
    contact_email: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class LocationTypeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class CampusBlockResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str
    name: str
    location_type_id: int
    capacity: int | None = None
    responsible_department_id: int | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class DepartmentListResponse(BaseModel):
    total: int
    departments: list[DepartmentResponse]


class LocationTypeListResponse(BaseModel):
    total: int
    location_types: list[LocationTypeResponse]


class CampusBlockListResponse(BaseModel):
    total: int
    campus_blocks: list[CampusBlockResponse]


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
    contact_email: str | None = Field(
        default=None,
        max_length=255,
    )


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
        ge=1,
        le=100000,
        examples=[240],
    )
    responsible_department_id: int | None = Field(
        default=None,
        ge=1,
        examples=[3],
    )


class DepartmentUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=150)
    description: str | None = Field(default=None, max_length=1000)
    contact_email: str | None = Field(default=None, max_length=255)
    is_active: bool | None = None


class CampusBlockUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=150)
    location_type_id: int | None = Field(default=None, ge=1)
    capacity: int | None = Field(default=None, ge=1, le=100000)
    responsible_department_id: int | None = Field(default=None, ge=1)
    is_active: bool | None = None

class ComplaintImpactVerificationRequest(BaseModel):
    verified_affected_population: int | None = Field(
        default=None,
        ge=1,
        le=100000,
        examples=[120],
    )

    impact_verification_status: str = Field(
        examples=["Verified"],
    )

    impact_verification_note: str | None = Field(
        default=None,
        max_length=1000,
    )