from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from apps.api.core.dependencies import get_current_user, require_roles
from apps.api.core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    verify_password,
)
from apps.api.schemas.auth import (
    LoginRequest,
    TokenResponse,
    UserResponse,
)
from apps.api.schemas.complaint import (
    ComplaintCreateResponse,
    ComplaintPredictionRequest,
    ComplaintPredictionResponse,
    ComplaintQueueResponse,
    ComplaintStatusUpdateRequest,
    DashboardSummaryResponse,
    StoredComplaintResponse,
)
from apps.api.schemas.feedback import (
    MLFeedbackResponse,
    MLFeedbackUpdateRequest,
)
from apps.api.schemas.history import (
    ComplaintStatusHistoryListResponse,
    ComplaintStatusHistoryResponse,
)
from apps.api.schemas.master_data import (
    CampusBlockCreateRequest,
    CampusBlockListResponse,
    CampusBlockResponse,
    CampusBlockUpdateRequest,
    ComplaintImpactVerificationRequest,
    DepartmentCreateRequest,
    DepartmentListResponse,
    DepartmentResponse,
    DepartmentUpdateRequest,
    LocationTypeListResponse,
    LocationTypeResponse,
)
from apps.api.services.prediction_service import PredictionService
from src.database.database import get_db, initialise_database
from src.database.models import Complaint, User
from src.database.repository import (
    count_complaints,
    create_audit_log,
    create_campus_block,
    create_complaint,
    create_complaint_ownership,
    create_department,
    create_status_history,
    get_campus_block_by_code,
    get_campus_block_by_id,
    get_campus_block_by_name,
    get_complaint_by_reference,
    get_dashboard_summary,
    get_department_by_code,
    get_department_by_id,
    get_department_by_name,
    get_location_type_by_id,
    get_or_create_complaint_verification,
    get_or_create_ml_feedback_record,
    get_user_by_email,
    is_complaint_owner,
    list_campus_blocks,
    list_complaints,
    list_departments,
    list_location_types,
    list_status_history,
    update_campus_block,
    update_complaint,
    update_complaint_verification,
    update_department,
    update_ml_feedback_record,
)

prediction_service = PredictionService()


def serialize_complaint(complaint: Complaint) -> StoredComplaintResponse:
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
    "/auth/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = get_user_by_email(
        db=db,
        email=str(request.email),
    )

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        subject=str(user.id),
        role=user.role,
    )

    return TokenResponse(
        access_token=access_token,
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES,
        user=UserResponse.model_validate(user),
    )


@app.get(
    "/auth/me",
    response_model=UserResponse,
)
def get_authenticated_user(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse.model_validate(current_user)


def can_access_complaint(
    db: Session,
    complaint: Complaint,
    current_user: User,
) -> bool:
    if current_user.role in {"Staff", "Admin"}:
        return True

    if current_user.role == "Student":
        return is_complaint_owner(
            db=db,
            complaint_id=complaint.id,
            user_id=current_user.id,
        )

    return False


@app.get(
    "/master-data/departments",
    response_model=DepartmentListResponse,
)
def get_departments(
    include_inactive: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> DepartmentListResponse:
    departments = list_departments(
        db=db,
        active_only=not include_inactive,
    )

    return DepartmentListResponse(
        total=len(departments),
        departments=[
            DepartmentResponse.model_validate(department)
            for department in departments
        ],
    )


@app.get(
    "/master-data/location-types",
    response_model=LocationTypeListResponse,
)
def get_location_types(
    include_inactive: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> LocationTypeListResponse:
    location_types = list_location_types(
        db=db,
        active_only=not include_inactive,
    )

    return LocationTypeListResponse(
        total=len(location_types),
        location_types=[
            LocationTypeResponse.model_validate(location_type)
            for location_type in location_types
        ],
    )


@app.get(
    "/master-data/campus-blocks",
    response_model=CampusBlockListResponse,
)
def get_campus_blocks(
    location_type_id: int | None = Query(default=None, ge=1),
    include_inactive: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> CampusBlockListResponse:
    campus_blocks = list_campus_blocks(
        db=db,
        location_type_id=location_type_id,
        active_only=not include_inactive,
    )

    return CampusBlockListResponse(
        total=len(campus_blocks),
        campus_blocks=[
            CampusBlockResponse.model_validate(campus_block)
            for campus_block in campus_blocks
        ],
    )


@app.post(
    "/master-data/departments",
    response_model=DepartmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_department(
    request: DepartmentCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> DepartmentResponse:
    code = request.code.strip().upper()
    name = request.name.strip()

    if get_department_by_code(db, code) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Department code already exists: {code}",
        )

    if get_department_by_name(db, name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Department name already exists: {name}",
        )

    try:
        department = create_department(
            db=db,
            data=request.model_dump(),
        )

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="CREATE_DEPARTMENT",
            entity_type="Department",
            entity_id=str(department.id),
            details=f"Created department {department.code}: {department.name}",
        )

        return DepartmentResponse.model_validate(department)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create department: {str(error)}",
        ) from error


@app.patch(
    "/master-data/departments/{department_id}",
    response_model=DepartmentResponse,
)
def edit_department(
    department_id: int,
    request: DepartmentUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> DepartmentResponse:
    department = get_department_by_id(db, department_id)

    if department is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department not found: {department_id}",
        )

    updates = request.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one department field to update.",
        )

    if "name" in updates and updates["name"] is not None:
        matching_department = get_department_by_name(
            db,
            str(updates["name"]).strip(),
        )

        if (
            matching_department is not None
            and matching_department.id != department_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Department name already exists.",
            )

    try:
        updated_department = update_department(
            db=db,
            department=department,
            updates=updates,
        )

        changed_fields = ", ".join(sorted(updates.keys()))

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="UPDATE_DEPARTMENT",
            entity_type="Department",
            entity_id=str(updated_department.id),
            details=(
                f"Updated department {updated_department.code}; "
                f"fields: {changed_fields}"
            ),
        )

        return DepartmentResponse.model_validate(updated_department)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not update department: {str(error)}",
        ) from error


@app.post(
    "/master-data/campus-blocks",
    response_model=CampusBlockResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_campus_block(
    request: CampusBlockCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Staff", "Admin")),
) -> CampusBlockResponse:
    code = request.code.strip().upper()
    name = request.name.strip()

    if get_campus_block_by_code(db, code) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Campus block code already exists: {code}",
        )

    if get_campus_block_by_name(db, name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Campus block name already exists: {name}",
        )

    location_type = get_location_type_by_id(
        db,
        request.location_type_id,
    )

    if location_type is None or not location_type.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected location type is unavailable.",
        )

    if request.responsible_department_id is not None:
        department = get_department_by_id(
            db,
            request.responsible_department_id,
        )

        if department is None or not department.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected responsible department is unavailable.",
            )

    try:
        campus_block = create_campus_block(
            db=db,
            data=request.model_dump(),
        )

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="CREATE_CAMPUS_BLOCK",
            entity_type="CampusBlock",
            entity_id=str(campus_block.id),
            details=(
                f"Created campus block {campus_block.code}: "
                f"{campus_block.name}"
            ),
        )

        return CampusBlockResponse.model_validate(campus_block)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create campus block: {str(error)}",
        ) from error


@app.patch(
    "/master-data/campus-blocks/{block_id}",
    response_model=CampusBlockResponse,
)
def edit_campus_block(
    block_id: int,
    request: CampusBlockUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> CampusBlockResponse:
    campus_block = get_campus_block_by_id(db, block_id)

    if campus_block is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campus block not found: {block_id}",
        )

    updates = request.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one campus-block field to update.",
        )

    if "location_type_id" in updates:
        location_type = get_location_type_by_id(
            db,
            int(updates["location_type_id"]),
        )

        if location_type is None or not location_type.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected location type is unavailable.",
            )

    if (
        "responsible_department_id" in updates
        and updates["responsible_department_id"] is not None
    ):
        department = get_department_by_id(
            db,
            int(updates["responsible_department_id"]),
        )

        if department is None or not department.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected responsible department is unavailable.",
            )

    try:
        updated_block = update_campus_block(
            db=db,
            campus_block=campus_block,
            updates=updates,
        )

        changed_fields = ", ".join(sorted(updates.keys()))

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="UPDATE_CAMPUS_BLOCK",
            entity_type="CampusBlock",
            entity_id=str(updated_block.id),
            details=(
                f"Updated campus block {updated_block.code}; "
                f"fields: {changed_fields}"
            ),
        )

        return CampusBlockResponse.model_validate(updated_block)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not update campus block: {str(error)}",
        ) from error


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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
    current_user: User = Depends(get_current_user),
) -> ComplaintCreateResponse:
    try:
        payload = request.model_dump()
        prediction = prediction_service.predict(payload)

        complaint = create_complaint(
            db=db,
            payload=payload,
            prediction=prediction,
        )

        create_complaint_ownership(
            db=db,
            complaint_id=complaint.id,
            submitted_by_user_id=current_user.id,
        )

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="CREATE_COMPLAINT",
            entity_type="Complaint",
            entity_id=str(complaint.id),
            details=(
                f"Complaint created with reference "
                f"{complaint.complaint_reference}"
            ),
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
    current_user: User = Depends(get_current_user),
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

    if current_user.role in {"Staff", "Admin"}:
        visible_complaints = complaints
        visible_total = total
    else:
        visible_complaints = [
            complaint
            for complaint in complaints
            if is_complaint_owner(
                db=db,
                complaint_id=complaint.id,
                user_id=current_user.id,
            )
        ]
        visible_total = len(visible_complaints)

    return ComplaintQueueResponse(
        total=visible_total,
        complaints=[
            serialize_complaint(complaint)
            for complaint in visible_complaints
        ],
    )


@app.get(
    "/complaints/{complaint_reference}",
    response_model=StoredComplaintResponse,
)
def get_stored_complaint(
    complaint_reference: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StoredComplaintResponse:
    complaint = get_complaint_by_reference(db, complaint_reference)

    if complaint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint not found: {complaint_reference}",
        )

    if not can_access_complaint(
        db=db,
        complaint=complaint,
        current_user=current_user,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this complaint.",
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
    current_user: User = Depends(require_roles("Staff", "Admin")),
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

    previous_status = complaint.status
    requested_status = updates.get("status")
    note_for_history = updates.get("staff_notes")

    updated_complaint = update_complaint(
        db=db,
        complaint=complaint,
        updates=updates,
    )

    if requested_status is not None and requested_status != previous_status:
        create_status_history(
            db=db,
            complaint_id=updated_complaint.id,
            old_status=previous_status,
            new_status=requested_status,
            changed_by_user_id=current_user.id,
            note=note_for_history,
        )

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="UPDATE_COMPLAINT_STATUS",
            entity_type="Complaint",
            entity_id=str(updated_complaint.id),
            details=(
                f"Status changed from {previous_status} "
                f"to {requested_status}"
            ),
        )
    elif "staff_notes" in updates:
        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="UPDATE_COMPLAINT_NOTES",
            entity_type="Complaint",
            entity_id=str(updated_complaint.id),
            details="Staff notes updated without a status change.",
        )

    return serialize_complaint(updated_complaint)


@app.get(
    "/complaints/{complaint_reference}/status-history",
    response_model=ComplaintStatusHistoryListResponse,
)
def get_complaint_status_history(
    complaint_reference: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ComplaintStatusHistoryListResponse:
    complaint = get_complaint_by_reference(db, complaint_reference)

    if complaint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint not found: {complaint_reference}",
        )

    if not can_access_complaint(
        db=db,
        complaint=complaint,
        current_user=current_user,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this complaint.",
        )

    history = list_status_history(
        db=db,
        complaint_id=complaint.id,
    )

    return ComplaintStatusHistoryListResponse(
        complaint_reference=complaint.complaint_reference,
        total=len(history),
        history=[
            ComplaintStatusHistoryResponse(
                id=item.id,
                old_status=item.old_status,
                new_status=item.new_status,
                note=item.note,
                changed_by_user_id=item.changed_by_user_id,
                changed_at=item.changed_at,
            )
            for item in history
        ],
    )


@app.patch(
    "/complaints/{complaint_reference}/impact-verification",
)
def verify_complaint_impact(
    complaint_reference: str,
    request: ComplaintImpactVerificationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Staff", "Admin")),
) -> dict[str, object]:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

    if complaint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint not found: {complaint_reference}",
        )

    verification = get_or_create_complaint_verification(
        db=db,
        complaint=complaint,
    )

    updates = request.model_dump(exclude_unset=True)

    if request.impact_verification_status == "Verified":
        if request.verified_affected_population is None:
            updates["verified_affected_population"] = (
                verification.reported_affected_population
            )

    if request.impact_verification_status == "Adjusted":
        if request.verified_affected_population is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "verified_affected_population is required when "
                    "impact_verification_status is Adjusted."
                ),
            )

    if request.impact_verification_status == "Disputed":
        updates["verified_affected_population"] = None

    updated_verification = update_complaint_verification(
        db=db,
        verification=verification,
        updates=updates,
        verified_by_user_id=current_user.id,
    )

    create_audit_log(
        db=db,
        actor_user_id=current_user.id,
        action="VERIFY_COMPLAINT_IMPACT",
        entity_type="Complaint",
        entity_id=str(complaint.id),
        details=(
            f"Impact verification set to "
            f"{updated_verification.impact_verification_status}; "
            f"reported={updated_verification.reported_affected_population}; "
            f"verified={updated_verification.verified_affected_population}"
        ),
    )

    return {
        "complaint_reference": complaint.complaint_reference,
        "reported_affected_population": (
            updated_verification.reported_affected_population
        ),
        "verified_affected_population": (
            updated_verification.verified_affected_population
        ),
        "impact_verification_status": (
            updated_verification.impact_verification_status
        ),
        "impact_verification_note": (
            updated_verification.impact_verification_note
        ),
        "verified_by_user_id": updated_verification.verified_by_user_id,
        "verified_at": updated_verification.verified_at,
    }


@app.patch(
    "/complaints/{complaint_reference}/ml-feedback",
    response_model=MLFeedbackResponse,
)
def update_complaint_ml_feedback(
    complaint_reference: str,
    request: MLFeedbackUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Staff", "Admin")),
) -> MLFeedbackResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

    if complaint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint not found: {complaint_reference}",
        )

    updates = request.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one ML feedback field to update.",
        )

    if "final_department_id" in updates:
        department_id = updates["final_department_id"]

        if department_id is not None:
            department = get_department_by_id(
                db=db,
                department_id=int(department_id),
            )

            if department is None or not department.is_active:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Selected final department is unavailable.",
                )

    if updates.get("training_eligible") is True:
        if complaint.status not in {"Resolved", "Closed"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Complaint must be Resolved or Closed before it can be "
                    "marked training eligible."
                ),
            )

        required_fields = [
            "final_category",
            "final_department_id",
            "final_priority",
            "actual_resolution_hours",
            "duplicate_decision",
        ]

        existing_feedback = get_or_create_ml_feedback_record(
            db=db,
            complaint=complaint,
        )

        combined_values = {
            field_name: updates.get(
                field_name,
                getattr(existing_feedback, field_name),
            )
            for field_name in required_fields
        }

        missing_fields = [
            field_name
            for field_name, value in combined_values.items()
            if value is None
        ]

        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Cannot mark training eligible. Missing verified fields: "
                    f"{missing_text}"
                ),
            )

    feedback = get_or_create_ml_feedback_record(
        db=db,
        complaint=complaint,
    )

    updated_feedback = update_ml_feedback_record(
        db=db,
        feedback=feedback,
        updates=updates,
        reviewed_by_user_id=current_user.id,
    )

    create_audit_log(
        db=db,
        actor_user_id=current_user.id,
        action="UPDATE_ML_FEEDBACK",
        entity_type="Complaint",
        entity_id=str(complaint.id),
        details=(
            f"ML feedback updated; "
            f"training_eligible={updated_feedback.training_eligible}"
        ),
    )

    return MLFeedbackResponse(
        complaint_reference=complaint.complaint_reference,
        final_category=updated_feedback.final_category,
        final_department_id=updated_feedback.final_department_id,
        final_priority=updated_feedback.final_priority,
        actual_resolution_hours=updated_feedback.actual_resolution_hours,
        duplicate_decision=updated_feedback.duplicate_decision,
        training_eligible=updated_feedback.training_eligible,
        exclusion_reason=updated_feedback.exclusion_reason,
        reviewed_by_user_id=updated_feedback.reviewed_by_user_id,
        reviewed_at=updated_feedback.reviewed_at,
    )


@app.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
def dashboard_summary(
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:
    return DashboardSummaryResponse(**get_dashboard_summary(db))