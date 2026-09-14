from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./campusresolve.db",
)

connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def initialise_database() -> None:
    from src.database.models import (
    AuditLog,
    CampusBlock,
    Complaint,
    ComplaintOwnership,
    ComplaintStatusHistory,
    ComplaintVerification,
    Department,
    LocationType,
    MLFeedbackRecord,
    User,
)

    Base.metadata.create_all(bind=engine)

def create_audit_log(
    db: Session,
    actor_user_id: int | None,
    action: str,
    entity_type: str,
    entity_id: str,
    details: str | None = None,
) -> AuditLog:
    audit_log = AuditLog(
        actor_user_id=actor_user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return audit_log


def list_audit_logs(
    db: Session,
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 100,
) -> list[AuditLog]:
    query = select(AuditLog).order_by(desc(AuditLog.created_at))

    filters = []

    if entity_type:
        filters.append(AuditLog.entity_type == entity_type)

    if entity_id:
        filters.append(AuditLog.entity_id == entity_id)

    if filters:
        query = query.where(and_(*filters))

    return list(db.scalars(query.limit(limit)).all())