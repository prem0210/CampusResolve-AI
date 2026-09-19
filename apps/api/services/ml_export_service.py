from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPORT_DIRECTORY = Path("data/exports")
SCHEMA_VERSION = "1.0"


def export_training_feedback_dataset(
    records: list[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    exported_at = datetime.now(timezone.utc)
    timestamp = exported_at.strftime("%Y%m%dT%H%M%SZ")
    dataset_version = f"training-feedback-{timestamp}"

    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    csv_path = EXPORT_DIRECTORY / f"{dataset_version}.csv"
    manifest_path = EXPORT_DIRECTORY / f"{dataset_version}.json"

    fieldnames = [
        "complaint_id",
        "complaint_reference",
        "complaint_text",
        "language",
        "location_type",
        "specific_location",
        "reported_affected_population",
        "verified_affected_population",
        "impact_verification_status",
        "final_category",
        "final_department_id",
        "final_priority",
        "actual_resolution_hours",
        "duplicate_decision",
        "complaint_status",
        "complaint_created_at",
        "feedback_reviewed_at",
        "feedback_reviewed_by_user_id",
    ]

    exported_rows: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []

    for complaint, feedback, verification in records:
        if verification is None:
            excluded_records.append(
                {
                    "complaint_reference": complaint.complaint_reference,
                    "reason": "Missing impact-verification record.",
                }
            )
            continue

        if verification.impact_verification_status not in {
            "Verified",
            "Adjusted",
        }:
            excluded_records.append(
                {
                    "complaint_reference": complaint.complaint_reference,
                    "reason": (
                        "Impact verification must be Verified or Adjusted; "
                        f"received {verification.impact_verification_status}."
                    ),
                }
            )
            continue

        if verification.verified_affected_population is None:
            excluded_records.append(
                {
                    "complaint_reference": complaint.complaint_reference,
                    "reason": "Verified affected population is missing.",
                }
            )
            continue

        exported_rows.append(
            {
                "complaint_id": complaint.id,
                "complaint_reference": complaint.complaint_reference,
                "complaint_text": complaint.complaint_text,
                "language": complaint.language,
                "location_type": complaint.location_type,
                "specific_location": complaint.specific_location,
                "reported_affected_population": (
                    verification.reported_affected_population
                ),
                "verified_affected_population": (
                    verification.verified_affected_population
                ),
                "impact_verification_status": (
                    verification.impact_verification_status
                ),
                "final_category": feedback.final_category,
                "final_department_id": feedback.final_department_id,
                "final_priority": feedback.final_priority,
                "actual_resolution_hours": feedback.actual_resolution_hours,
                "duplicate_decision": feedback.duplicate_decision,
                "complaint_status": complaint.status,
                "complaint_created_at": complaint.created_at.isoformat(),
                "feedback_reviewed_at": (
                    feedback.reviewed_at.isoformat()
                    if feedback.reviewed_at is not None
                    else None
                ),
                "feedback_reviewed_by_user_id": feedback.reviewed_by_user_id,
            }
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(exported_rows)

    sha256 = hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()

    manifest = {
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "exported_at": exported_at.isoformat(),
        "record_count": len(exported_rows),
        "candidate_record_count": len(records),
        "excluded_record_count": len(excluded_records),
        "excluded_records": excluded_records,
        "sha256": sha256,
        "csv_file": str(csv_path),
        "selection_rules": {
            "training_eligible": True,
            "complaint_status": ["Resolved", "Closed"],
            "impact_verification_status": ["Verified", "Adjusted"],
            "verified_affected_population_required": True,
        },
        "columns": fieldnames,
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "dataset_version": dataset_version,
        "exported_at": exported_at,
        "record_count": len(exported_rows),
        "csv_file": str(csv_path),
        "manifest_file": str(manifest_path),
        "sha256": sha256,
    }