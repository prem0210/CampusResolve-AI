from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./campusresolve.db",
).strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL cannot be empty.")


is_sqlite = DATABASE_URL.startswith("sqlite")

connect_args: dict[str, bool] = (
    {"check_same_thread": False}
    if is_sqlite
    else {}
)


engine: Engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=not is_sqlite,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialise_database() -> None:
    # Import models before create_all() so SQLAlchemy registers every table
    # on Base.metadata.
    from src.database.models import (
        AuditLog,
        CampusBlock,
        Complaint,
        ComplaintAssignment,
        ComplaintAssignmentHistory,
        ComplaintEscalation,
        ComplaintOwnership,
        ComplaintStatusHistory,
        ComplaintVerification,
        Department,
        LocationType,
        MLFeedbackRecord,
        User,
    )

    Base.metadata.create_all(bind=engine)