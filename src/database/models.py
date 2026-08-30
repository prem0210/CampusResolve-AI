from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.database import Base


class Complaint(Base):
    __tablename__ = "complaints"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    complaint_reference: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        index=True,
        nullable=False,
    )

    complaint_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False)

    location_type: Mapped[str] = mapped_column(String(100), nullable=False)
    specific_location: Mapped[str] = mapped_column(String(150), nullable=False)

    affected_population: Mapped[int] = mapped_column(Integer, nullable=False)
    safety_flag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    repeat_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    predicted_category: Mapped[str] = mapped_column(String(100), nullable=False)
    category_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    assigned_department: Mapped[str] = mapped_column(String(150), nullable=False)

    predicted_priority: Mapped[str] = mapped_column(String(20), nullable=False)
    priority_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    estimated_resolution_hours: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_interval_plus_minus_hours: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    duplicate_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    possible_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    top_duplicate_id: Mapped[str | None] = mapped_column(String(30), nullable=True)
    top_duplicate_similarity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="Open",
        nullable=False,
        index=True,
    )

    staff_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )