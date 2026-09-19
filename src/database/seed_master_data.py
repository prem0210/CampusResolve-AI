from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.database import SessionLocal, initialise_database
from src.database.models import CampusBlock, Department, LocationType, User

from apps.api.core.security import hash_password


DEPARTMENTS = [
    {
        "code": "MAINT",
        "name": "Maintenance Department",
        "description": "Handles plumbing, electrical, civil, and infrastructure issues.",
        "contact_email": None,
    },
    {
        "code": "IT",
        "name": "Information Technology Department",
        "description": "Handles Wi-Fi, network, computer, and digital service issues.",
        "contact_email": None,
    },
    {
        "code": "HOSTEL",
        "name": "Hostel Administration",
        "description": "Handles hostel administration, accommodation, and resident concerns.",
        "contact_email": None,
    },
    {
        "code": "HOUSEKEEPING",
        "name": "Housekeeping Department",
        "description": "Handles cleaning, washroom hygiene, waste, and sanitation issues.",
        "contact_email": None,
    },
    {
        "code": "SECURITY",
        "name": "Security Department",
        "description": "Handles campus safety, access, parking, and security concerns.",
        "contact_email": None,
    },
    {
        "code": "ADMIN",
        "name": "Campus Administration",
        "description": "Handles general campus, academic office, and administrative concerns.",
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


CAMPUS_BLOCKS = [
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

DEMO_USERS = [
    {
        "full_name": "System Administrator",
        "email": "admin@campusresolve.example.com",
        "password": "Admin@123",
        "role": "Admin",
        "department_code": None,
    },
    {
        "full_name": "Maintenance Staff",
        "email": "maintenance.staff@campusresolve.example.com",
        "password": "Staff@123",
        "role": "Staff",
        "department_code": "MAINT",
    },
    {
        "full_name": "Demo Student",
        "email": "student@campusresolve.example.com",
        "password": "Student@123",
        "role": "Student",
        "department_code": None,
    },
]

def get_or_create_department(
    db: Session,
    department_data: dict[str, str | None],
) -> Department:
    department = db.scalar(
        select(Department).where(Department.code == department_data["code"])
    )

    if department is None:
        department = Department(**department_data)
        db.add(department)
        db.flush()

    return department


def get_or_create_location_type(db: Session, name: str) -> LocationType:
    location_type = db.scalar(
        select(LocationType).where(LocationType.name == name)
    )

    if location_type is None:
        location_type = LocationType(name=name)
        db.add(location_type)
        db.flush()

    return location_type


def get_or_create_campus_block(
    db: Session,
    block_data: dict[str, str | int],
    location_types: dict[str, LocationType],
    departments: dict[str, Department],
) -> CampusBlock:
    campus_block = db.scalar(
        select(CampusBlock).where(CampusBlock.code == block_data["code"])
    )

    if campus_block is None:
        campus_block = CampusBlock(
            code=str(block_data["code"]),
            name=str(block_data["name"]),
            location_type_id=location_types[
                str(block_data["location_type"])
            ].id,
            capacity=int(block_data["capacity"]),
            responsible_department_id=departments[
                str(block_data["department_code"])
            ].id,
        )
        db.add(campus_block)
        db.flush()

    return campus_block

def get_or_create_user(
    db: Session,
    user_data: dict[str, str | None],
    departments: dict[str, Department],
) -> User:
    user = db.scalar(
        select(User).where(User.email == user_data["email"])
    )

    if user is None:
        department_code = user_data["department_code"]

        user = User(
            full_name=str(user_data["full_name"]),
            email=str(user_data["email"]).lower(),
            password_hash=hash_password(str(user_data["password"])),
            role=str(user_data["role"]),
            department_id=(
                departments[str(department_code)].id
                if department_code
                else None
            ),
        )
        db.add(user)
        db.flush()

    return user

def main() -> None:
    initialise_database()

    db = SessionLocal()

    try:
        departments: dict[str, Department] = {}

        for department_data in DEPARTMENTS:
            department = get_or_create_department(db, department_data)
            departments[department.code] = department

        location_types: dict[str, LocationType] = {}

        for location_type_name in LOCATION_TYPES:
            location_type = get_or_create_location_type(db, location_type_name)
            location_types[location_type.name] = location_type

        for block_data in CAMPUS_BLOCKS:
            get_or_create_campus_block(
                db=db,
                block_data=block_data,
                location_types=location_types,
                departments=departments,
            )
        
        for user_data in DEMO_USERS:
            get_or_create_user(
                db=db,
                user_data=user_data,
                departments=departments,
            )

        db.commit()

        print("Master data seeded successfully.")
        print(f"Departments: {len(DEPARTMENTS)}")
        print(f"Location types: {len(LOCATION_TYPES)}")
        print(f"Campus blocks: {len(CAMPUS_BLOCKS)}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()