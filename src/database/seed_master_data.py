from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.core.security import hash_password
from src.database.database import SessionLocal, initialise_database
from src.database.models import CampusBlock, Department, LocationType, User


DEPARTMENTS: list[dict[str, str | None]] = [
    {
        "code": "MAINT",
        "name": "Maintenance Department",
        "description": (
            "Handles plumbing, electrical, civil, and infrastructure issues."
        ),
        "contact_email": None,
    },
    {
        "code": "IT",
        "name": "Information Technology Department",
        "description": (
            "Handles Wi-Fi, network, computer, and digital service issues."
        ),
        "contact_email": None,
    },
    {
        "code": "HOSTEL",
        "name": "Hostel Administration",
        "description": (
            "Handles hostel administration, accommodation, and resident concerns."
        ),
        "contact_email": None,
    },
    {
        "code": "HOUSEKEEPING",
        "name": "Housekeeping Department",
        "description": (
            "Handles cleaning, washroom hygiene, waste, and sanitation issues."
        ),
        "contact_email": None,
    },
    {
        "code": "SECURITY",
        "name": "Security Department",
        "description": (
            "Handles campus safety, access, parking, and security concerns."
        ),
        "contact_email": None,
    },
    {
        "code": "ADMIN",
        "name": "Campus Administration",
        "description": (
            "Handles general campus, academic office, and administrative concerns."
        ),
        "contact_email": None,
    },
]


LOCATION_TYPES = [
    "Hostel",
    "Classroom",
    "Laboratory",
    "Library",
    "Washroom",
    "Canteen",
    "Campus Building",
    "Campus Road",
    "Parking",
    "Office",
]


CAMPUS_BLOCKS: list[dict[str, str | int]] = [
    {
        "code": "HB-A",
        "name": "Hostel Block A",
        "location_type": "Hostel",
        "capacity": 240,
        "department_code": "HOSTEL",
    },
    {
        "code": "HB-B",
        "name": "Hostel Block B",
        "location_type": "Hostel",
        "capacity": 240,
        "department_code": "HOSTEL",
    },
    {
        "code": "HB-C",
        "name": "Hostel Block C",
        "location_type": "Hostel",
        "capacity": 240,
        "department_code": "HOSTEL",
    },
    {
        "code": "LIB-01",
        "name": "Central Library",
        "location_type": "Library",
        "capacity": 500,
        "department_code": "ADMIN",
    },
    {
        "code": "C-204",
        "name": "Classroom C-204",
        "location_type": "Classroom",
        "capacity": 60,
        "department_code": "MAINT",
    },
    {
        "code": "LAB-01",
        "name": "Computer Laboratory 1",
        "location_type": "Laboratory",
        "capacity": 60,
        "department_code": "IT",
    },
    {
        "code": "CAF-01",
        "name": "Main Canteen",
        "location_type": "Canteen",
        "capacity": 300,
        "department_code": "HOUSEKEEPING",
    },
    {
        "code": "PK-01",
        "name": "Main Parking Area",
        "location_type": "Parking",
        "capacity": 400,
        "department_code": "SECURITY",
    },
    {
        "code": "ADMIN-01",
        "name": "Administrative Block",
        "location_type": "Office",
        "capacity": 200,
        "department_code": "ADMIN",
    },
]


def environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()

    if not value:
        return default

    return value in {"1", "true", "yes", "on"}


def get_or_create_department(
    db: Session,
    department_data: Mapping[str, str | None],
) -> tuple[Department, bool]:
    code = str(department_data["code"]).strip().upper()

    department = db.scalar(
        select(Department).where(Department.code == code)
    )

    if department is not None:
        return department, False

    department = Department(
        code=code,
        name=str(department_data["name"]).strip(),
        description=department_data["description"],
        contact_email=(
            str(department_data["contact_email"]).strip().lower()
            if department_data["contact_email"]
            else None
        ),
        is_active=True,
    )

    db.add(department)
    db.flush()

    return department, True


def get_or_create_location_type(
    db: Session,
    name: str,
) -> tuple[LocationType, bool]:
    normalized_name = name.strip()

    location_type = db.scalar(
        select(LocationType).where(
            LocationType.name == normalized_name
        )
    )

    if location_type is not None:
        return location_type, False

    location_type = LocationType(
        name=normalized_name,
        is_active=True,
    )

    db.add(location_type)
    db.flush()

    return location_type, True


def get_or_create_campus_block(
    db: Session,
    block_data: Mapping[str, str | int],
    location_types: Mapping[str, LocationType],
    departments: Mapping[str, Department],
) -> tuple[CampusBlock, bool]:
    code = str(block_data["code"]).strip().upper()

    campus_block = db.scalar(
        select(CampusBlock).where(CampusBlock.code == code)
    )

    if campus_block is not None:
        return campus_block, False

    location_type_name = str(block_data["location_type"]).strip()
    department_code = str(block_data["department_code"]).strip().upper()

    campus_block = CampusBlock(
        code=code,
        name=str(block_data["name"]).strip(),
        location_type_id=location_types[location_type_name].id,
        capacity=int(block_data["capacity"]),
        responsible_department_id=departments[department_code].id,
        is_active=True,
    )

    db.add(campus_block)
    db.flush()

    return campus_block, True


def build_demo_users() -> list[dict[str, str | None]]:
    admin_password = os.getenv("SEED_ADMIN_PASSWORD", "").strip()
    staff_password = os.getenv("SEED_STAFF_PASSWORD", "").strip()
    student_password = os.getenv("SEED_STUDENT_PASSWORD", "").strip()

    required_passwords = {
        "SEED_ADMIN_PASSWORD": admin_password,
        "SEED_STAFF_PASSWORD": staff_password,
        "SEED_STUDENT_PASSWORD": student_password,
    }

    missing_names = [
        name
        for name, password in required_passwords.items()
        if not password
    ]

    if missing_names:
        missing_text = ", ".join(missing_names)
        raise RuntimeError(
            "Demo-user seeding requires these environment variables: "
            f"{missing_text}"
        )

    return [
        {
            "full_name": "System Administrator",
            "email": "admin@campusresolve.example.com",
            "password": admin_password,
            "role": "Admin",
            "department_code": None,
        },
        {
            "full_name": "Maintenance Staff",
            "email": "maintenance.staff@campusresolve.example.com",
            "password": staff_password,
            "role": "Staff",
            "department_code": "MAINT",
        },
        {
            "full_name": "Demo Student",
            "email": "student@campusresolve.example.com",
            "password": student_password,
            "role": "Student",
            "department_code": None,
        },
    ]


def get_or_create_user(
    db: Session,
    user_data: Mapping[str, str | None],
    departments: Mapping[str, Department],
) -> tuple[User, bool]:
    email = str(user_data["email"]).strip().lower()

    user = db.scalar(
        select(User).where(User.email == email)
    )

    if user is not None:
        return user, False

    department_code = user_data["department_code"]

    user = User(
        full_name=str(user_data["full_name"]).strip(),
        email=email,
        password_hash=hash_password(str(user_data["password"])),
        role=str(user_data["role"]).strip(),
        department_id=(
            departments[str(department_code).strip().upper()].id
            if department_code
            else None
        ),
        is_active=True,
    )

    db.add(user)
    db.flush()

    return user, True


def main() -> None:
    initialise_database()

    summary = {
        "departments_created": 0,
        "location_types_created": 0,
        "campus_blocks_created": 0,
        "demo_users_created": 0,
    }

    with SessionLocal.begin() as db:
        departments: dict[str, Department] = {}

        for department_data in DEPARTMENTS:
            department, created = get_or_create_department(
                db=db,
                department_data=department_data,
            )
            departments[department.code] = department
            summary["departments_created"] += int(created)

        location_types: dict[str, LocationType] = {}

        for location_type_name in LOCATION_TYPES:
            location_type, created = get_or_create_location_type(
                db=db,
                name=location_type_name,
            )
            location_types[location_type.name] = location_type
            summary["location_types_created"] += int(created)

        for block_data in CAMPUS_BLOCKS:
            _, created = get_or_create_campus_block(
                db=db,
                block_data=block_data,
                location_types=location_types,
                departments=departments,
            )
            summary["campus_blocks_created"] += int(created)

        if environment_flag("SEED_DEMO_USERS"):
            for user_data in build_demo_users():
                _, created = get_or_create_user(
                    db=db,
                    user_data=user_data,
                    departments=departments,
                )
                summary["demo_users_created"] += int(created)

    print("Master data seeding completed.")
    print(
        f"Departments created: {summary['departments_created']} "
        f"(configured: {len(DEPARTMENTS)})"
    )
    print(
        f"Location types created: {summary['location_types_created']} "
        f"(configured: {len(LOCATION_TYPES)})"
    )
    print(
        f"Campus blocks created: {summary['campus_blocks_created']} "
        f"(configured: {len(CAMPUS_BLOCKS)})"
    )
    print(
        f"Demo users created: {summary['demo_users_created']}"
    )


if __name__ == "__main__":
    main()