from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class MLTrainingExportResponse(BaseModel):
    dataset_version: str
    exported_at: datetime
    record_count: int
    csv_file: str
    manifest_file: str
    sha256: str