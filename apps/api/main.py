from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from apps.api.schemas.complaint import (
    ComplaintCreateResponse,
    ComplaintPredictionRequest,
    ComplaintPredictionResponse,
    ComplaintQueueResponse,
    ComplaintStatusUpdateRequest,
    DashboardSummaryResponse,
    StoredComplaintResponse,
)
from apps.api.services.prediction_service import PredictionService
from src.database.database import get_db, initialise_database
from src.database.repository import (
    count_complaints,
    create_complaint,
    get_complaint_by_reference,
    get_dashboard_summary,
    list_complaints,
    update_complaint,
)

prediction_service = PredictionService()

def serialize_complaint(complaint) -> StoredComplaintResponse:
    return StoredComplaintResponse(
        complaint_reference=complaint.complaint_reference,
        complaint_text=complaint.complaint_text,
        language=complaint.language,
        location_type=complaint.location_type,
        specific_location=complaint.specific_location,
        affected_population=complaint.affected_population,
        safety_flag=complaint.safety_flag,
        repeat_count=complaint.repeat_count,
        predicted_category=complaint.predicted_category,
        category_confidence=complaint.category_confidence,
        assigned_department=complaint.assigned_department,
        predicted_priority=complaint.predicted_priority,
        priority_confidence=complaint.priority_confidence,
        estimated_resolution_hours=complaint.estimated_resolution_hours,
        prediction_interval_plus_minus_hours=(
            complaint.prediction_interval_plus_minus_hours
        ),
        possible_duplicate=complaint.possible_duplicate,
        top_duplicate_id=complaint.top_duplicate_id,
        top_duplicate_similarity=complaint.top_duplicate_similarity,
        status=complaint.status,
        staff_notes=complaint.staff_notes,
        created_at=complaint.created_at,
        updated_at=complaint.updated_at,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing CampusResolve-AI database...")
    initialise_database()

    print("Loading CampusResolve-AI model artifacts...")
    prediction_service.load_artifacts()

    print("CampusResolve-AI prediction service is ready.")
    yield
    print("CampusResolve-AI API is shutting down.")


app = FastAPI(
    title="CampusResolve-AI API",
    description=(
        "Multilingual campus complaint intelligence API for classification, "
        "routing, priority prediction, duplicate detection, "
        "resolution-time estimation, and explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "CampusResolve-AI API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check(
    db: Session = Depends(get_db),
) -> dict[str, str | bool | int]:
    return {
        "status": "healthy" if prediction_service.is_ready() else "starting",
        "service": "CampusResolve-AI",
        "models_loaded": prediction_service.is_ready(),
        "stored_complaints": count_complaints(db),
    }


@app.post(
    "/predict",
    response_model=ComplaintPredictionResponse,
)
def predict_complaint(
    request: ComplaintPredictionRequest,
) -> ComplaintPredictionResponse:
    try:
        result = prediction_service.predict(request.model_dump())
        return ComplaintPredictionResponse(**result)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error
@app.post(
    "/complaints",
    response_model=ComplaintCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_and_save_complaint(
    request: ComplaintPredictionRequest,
    db: Session = Depends(get_db),
) -> ComplaintCreateResponse:
    try:
        payload = request.model_dump()
        prediction = prediction_service.predict(payload)

        complaint = create_complaint(
            db=db,
            payload=payload,
            prediction=prediction,
        )

        return ComplaintCreateResponse(
            **prediction,
            complaint_reference=complaint.complaint_reference,
            status=complaint.status,
            created_at=complaint.created_at,
        )
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Complaint creation failed: {str(error)}",
        ) from error
@app.get(
    "/complaints",
    response_model=ComplaintQueueResponse,
)
def get_complaints(
    complaint_status: str | None = Query(default=None, alias="status"),
    priority: str | None = Query(default=None),
    department: str | None = Query(default=None),
    category: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ComplaintQueueResponse:
    total, complaints = list_complaints(
        db=db,
        status=complaint_status,
        priority=priority,
        department=department,
        category=category,
        limit=limit,
        offset=offset,
    )

    return ComplaintQueueResponse(
        total=total,
        complaints=[
            serialize_complaint(complaint)
            for complaint in complaints
        ],
    )


@app.get(
    "/complaints/{complaint_reference}",
    response_model=StoredComplaintResponse,
)
def get_complaint(
    complaint_reference: str,
    db: Session = Depends(get_db),
) -> StoredComplaintResponse:
    complaint = get_complaint_by_reference(db, complaint_reference)

    if complaint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint not found: {complaint_reference}",
        )

    return serialize_complaint(complaint)


@app.patch(
    "/complaints/{complaint_reference}",
    response_model=StoredComplaintResponse,
)
def update_stored_complaint(
    complaint_reference: str,
    request: ComplaintStatusUpdateRequest,
    db: Session = Depends(get_db),
) -> StoredComplaintResponse:
    complaint = get_complaint_by_reference(db, complaint_reference)

    if complaint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint not found: {complaint_reference}",
        )

    updates = request.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one field: status or staff_notes.",
        )

    updated_complaint = update_complaint(
        db=db,
        complaint=complaint,
        updates=updates,
    )

    return serialize_complaint(updated_complaint)


@app.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
def dashboard_summary(
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:
    return DashboardSummaryResponse(**get_dashboard_summary(db))