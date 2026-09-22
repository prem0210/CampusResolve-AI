from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from apps.api.core.dependencies import get_current_user, require_roles
from apps.api.core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    verify_password,
)
from apps.api.core.workflow import (
    ALLOWED_STATUS_TRANSITIONS,
    DUPLICATE_DECISIONS,
    ESCALATION_STATES,
    FINAL_PRIORITIES,
    IMPACT_VERIFICATION_STATUSES,
)
from apps.api.schemas.assignment import (
    ComplaintAssignmentRequest,
    ComplaintAssignmentResponse,
)
from apps.api.schemas.assignment_history import (
    ComplaintAssignmentHistoryEventResponse,
    ComplaintAssignmentHistoryListResponse,
)
from apps.api.schemas.audit import (
    AuditLogListResponse,
    AuditLogResponse,
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
from apps.api.schemas.escalation import (
    ComplaintEscalationResponse,
    ComplaintEscalationUpdateRequest,
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
from apps.api.schemas.ml_export import MLTrainingExportResponse
from apps.api.schemas.ml_monitoring import MLMonitoringResponse
from apps.api.schemas.ownership import (
    ComplaintOwnershipResponse,
    ComplaintOwnershipUpdateRequest,
    UnownedComplaintListResponse,
    UnownedComplaintResponse,
)
from apps.api.schemas.timeline import (
    ComplaintTimelineResponse,
    TimelineAssignmentResponse,
    TimelineEscalationResponse,
    TimelineImpactResponse,
    TimelineStatusEventResponse,
)
from apps.api.services.ml_export_service import (
    export_training_feedback_dataset,
)
from apps.api.services.prediction_service import PredictionService
from src.database.database import get_db, initialise_database
from src.database.models import Complaint, User
from src.database.repository import (
    can_staff_access_complaint,
    count_complaints,
    create_assignment_history,
    create_audit_log,
    create_campus_block,
    create_complaint,
    create_complaint_assignment,
    create_complaint_escalation,
    create_complaint_ownership,
    create_department,
    create_status_history,
    get_campus_block_by_code,
    get_campus_block_by_id,
    get_campus_block_by_name,
    get_complaint_assignment,
    get_complaint_by_reference,
    get_complaint_escalation,
    get_complaint_ownership,
    get_complaint_verification,
    get_dashboard_summary,
    get_department_by_code,
    get_department_by_id,
    get_department_by_name,
    get_location_type_by_id,
    get_ml_monitoring_summary,
    get_or_create_complaint_verification,
    get_or_create_ml_feedback_record,
    get_user_by_email,
    get_user_by_id,
    is_complaint_overdue,
    is_complaint_owner,
    list_assignment_history,
    list_audit_logs,
    list_campus_blocks,
    list_complaints,
    list_departments,
    list_location_types,
    list_status_history,
    list_training_eligible_records,
    list_unowned_complaints,
    update_campus_block,
    update_complaint,
    update_complaint_assignment,
    update_complaint_escalation,
    update_complaint_ownership,
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
        duplicate_threshold=complaint.duplicate_threshold,
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
    if current_user.role == "Admin":
        return True

    if current_user.role == "Student":
        return is_complaint_owner(
            db=db,
            complaint_id=complaint.id,
            user_id=current_user.id,
        )

    if current_user.role == "Staff":
        return can_staff_access_complaint(
            db=db,
            complaint_id=complaint.id,
            staff_user_id=current_user.id,
            staff_department_id=current_user.department_id,
        )

    return False


@app.get(
    "/master-data/departments",
    response_model=DepartmentListResponse,
)
def get_departments(
    include_inactive: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
) -> DepartmentListResponse:
    if include_inactive and current_user.role not in {"Staff", "Admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Staff or Admin users can view inactive departments.",
        )

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
    current_user: User | None = Depends(get_current_user),
) -> LocationTypeListResponse:
    if include_inactive and current_user.role not in {"Staff", "Admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Only Staff or Admin users can view inactive location types."
            ),
        )

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
    current_user: User | None = Depends(get_current_user),
) -> CampusBlockListResponse:
    if include_inactive and current_user.role not in {"Staff", "Admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Only Staff or Admin users can view inactive campus blocks."
            ),
        )

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

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(f"Department creation failed: {error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create department.",
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

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(f"Department update failed: {error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update department.",
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
                detail=(
                    "Selected responsible department is unavailable."
                ),
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

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(f"Campus-block creation failed: {error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create campus block.",
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
                detail=(
                    "Selected responsible department is unavailable."
                ),
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

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(f"Campus-block update failed: {error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update campus block.",
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

    except HTTPException:
        raise

    except Exception as error:
        print(f"Complaint prediction failed: {error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed.",
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

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(f"Complaint creation failed: {error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complaint could not be created.",
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
    assigned_to_me: bool = Query(default=False),
    assigned_to_user_id: int | None = Query(default=None, ge=1),
    assignment_state: str | None = Query(default=None),
    assigned_department_id: int | None = Query(default=None, ge=1),
    escalation_state: str | None = Query(default=None),
    due_before: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ComplaintQueueResponse:
    role = current_user.role.strip()

    if assignment_state not in {None, "Assigned", "Unassigned"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "assignment_state must be either Assigned or Unassigned."
            ),
        )

    if escalation_state not in {
        None,
        "OnTrack",
        "Escalated",
        "Overdue",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "escalation_state must be one of: "
                "OnTrack, Escalated, Overdue."
            ),
        )

    staff_queue_filters_requested = any(
        [
            assigned_to_me,
            assigned_to_user_id is not None,
            assignment_state is not None,
            assigned_department_id is not None,
            escalation_state is not None,
            due_before is not None,
        ]
    )

    if role == "Student" and staff_queue_filters_requested:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Assignment and escalation filters are available only "
                "to Staff and Admin."
            ),
        )

    if role == "Staff":
        if (
            assigned_to_user_id is not None
            and assigned_to_user_id != current_user.id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Staff users can filter assignments only for "
                    "their own account."
                ),
            )

        if (
            assigned_department_id is not None
            and assigned_department_id != current_user.department_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Staff users can filter assignments only for "
                    "their own department."
                ),
            )

    if assigned_to_me:
        assigned_to_user_id = current_user.id

    submitted_by_user_id = (
        current_user.id
        if role == "Student"
        else None
    )

    staff_user_id = (
        current_user.id
        if role == "Staff"
        else None
    )

    staff_department_id = (
        current_user.department_id
        if role == "Staff"
        else None
    )

    total, complaints = list_complaints(
        db=db,
        status=complaint_status,
        priority=priority,
        department=department,
        category=category,
        submitted_by_user_id=submitted_by_user_id,
        staff_user_id=staff_user_id,
        staff_department_id=staff_department_id,
        assigned_to_user_id=assigned_to_user_id,
        assignment_state=assignment_state,
        assigned_department_id=assigned_department_id,
        escalation_state=escalation_state,
        due_before=due_before,
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

    if not can_access_complaint(
        db=db,
        complaint=complaint,
        current_user=current_user,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this complaint.",
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

    if requested_status is not None:
        if requested_status == previous_status:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Complaint status is already set to this value.",
            )

        allowed_next_statuses = ALLOWED_STATUS_TRANSITIONS.get(
            previous_status,
            set(),
        )

        if requested_status not in allowed_next_statuses:
            allowed_text = ", ".join(sorted(allowed_next_statuses))

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid status transition from {previous_status} "
                    f"to {requested_status}. "
                    f"Allowed next status values: "
                    f"{allowed_text or 'none'}."
                ),
            )

    try:
        updated_complaint = update_complaint(
            db=db,
            complaint=complaint,
            updates=updates,
        )

        if (
            requested_status is not None
            and requested_status != previous_status
        ):
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

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(
            f"Complaint update failed for "
            f"{complaint_reference}: {error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complaint could not be updated.",
        ) from error

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


@app.get(
    "/complaints/{complaint_reference}/timeline",
    response_model=ComplaintTimelineResponse,
)
def get_complaint_timeline(
    complaint_reference: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ComplaintTimelineResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

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

    assignment = get_complaint_assignment(
        db=db,
        complaint_id=complaint.id,
    )

    verification = get_complaint_verification(
        db=db,
        complaint_id=complaint.id,
    )

    escalation = get_complaint_escalation(
        db=db,
        complaint_id=complaint.id,
    )

    history = list_status_history(
        db=db,
        complaint_id=complaint.id,
    )

    timeline_assignment = None

    if assignment is not None:
        timeline_assignment = TimelineAssignmentResponse(
            assigned_to_user_id=assignment.assigned_to_user_id,
            assigned_department_id=assignment.assigned_department_id,
            assignment_note=assignment.assignment_note,
            assigned_at=assignment.assigned_at,
        )

    timeline_impact = None

    if verification is not None:
        timeline_impact = TimelineImpactResponse(
            reported_affected_population=(
                verification.reported_affected_population
            ),
            verified_affected_population=(
                verification.verified_affected_population
            ),
            impact_verification_status=(
                verification.impact_verification_status
            ),
            impact_verification_note=(
                verification.impact_verification_note
            ),
            verified_at=verification.verified_at,
        )

    timeline_escalation = None

    if escalation is not None:
        timeline_escalation = TimelineEscalationResponse(
            due_at=escalation.due_at,
            escalation_state=escalation.escalation_state,
            is_overdue=is_complaint_overdue(
                complaint=complaint,
                escalation=escalation,
            ),
            escalated_at=escalation.escalated_at,
            escalation_reason=escalation.escalation_reason,
        )

    return ComplaintTimelineResponse(
        complaint_reference=complaint.complaint_reference,
        complaint_text=complaint.complaint_text,
        location_type=complaint.location_type,
        specific_location=complaint.specific_location,
        status=complaint.status,
        created_at=complaint.created_at,
        updated_at=complaint.updated_at,
        assignment=timeline_assignment,
        impact=timeline_impact,
        escalation=timeline_escalation,
        status_history=[
            TimelineStatusEventResponse(
                old_status=item.old_status,
                new_status=item.new_status,
                note=item.note,
                changed_at=item.changed_at,
            )
            for item in history
        ],
    )


@app.get(
    "/complaints/{complaint_reference}/assignment-history",
    response_model=ComplaintAssignmentHistoryListResponse,
)
def get_assignment_history(
    complaint_reference: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ComplaintAssignmentHistoryListResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

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

    history = list_assignment_history(
        db=db,
        complaint_id=complaint.id,
    )

    return ComplaintAssignmentHistoryListResponse(
        complaint_reference=complaint.complaint_reference,
        total=len(history),
        history=[
            ComplaintAssignmentHistoryEventResponse(
                previous_assigned_to_user_id=(
                    item.previous_assigned_to_user_id
                ),
                new_assigned_to_user_id=item.new_assigned_to_user_id,
                previous_department_id=item.previous_department_id,
                new_department_id=item.new_department_id,
                note=item.note,
                changed_at=item.changed_at,
            )
            for item in history
        ],
    )


@app.get(
    "/complaints/{complaint_reference}/assignment",
    response_model=ComplaintAssignmentResponse,
)
def get_assignment(
    complaint_reference: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ComplaintAssignmentResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

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

    assignment = get_complaint_assignment(
        db=db,
        complaint_id=complaint.id,
    )

    if assignment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No staff assignment exists for this complaint.",
        )

    return ComplaintAssignmentResponse(
        complaint_reference=complaint.complaint_reference,
        assigned_to_user_id=assignment.assigned_to_user_id,
        assigned_department_id=assignment.assigned_department_id,
        assignment_note=assignment.assignment_note,
        assigned_by_user_id=assignment.assigned_by_user_id,
        assigned_at=assignment.assigned_at,
        updated_at=assignment.updated_at,
    )


@app.put(
    "/complaints/{complaint_reference}/assignment",
    response_model=ComplaintAssignmentResponse,
)
def set_assignment(
    complaint_reference: str,
    request: ComplaintAssignmentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Staff", "Admin")),
) -> ComplaintAssignmentResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

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

    assigned_user = get_user_by_id(
        db=db,
        user_id=request.assigned_to_user_id,
    )

    if (
        assigned_user is None
        or not assigned_user.is_active
        or assigned_user.role not in {"Staff", "Admin"}
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assigned user must be an active Staff or Admin user.",
        )

    if request.assigned_department_id is not None:
        department = get_department_by_id(
            db=db,
            department_id=request.assigned_department_id,
        )

        if department is None or not department.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assigned department is unavailable.",
            )

    if current_user.role == "Staff":
        if request.assigned_to_user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Staff users can assign a complaint only to "
                    "their own account."
                ),
            )

        if current_user.department_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Staff users must belong to a department before "
                    "they can assign complaints."
                ),
            )

        if request.assigned_department_id != current_user.department_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Staff users can assign complaints only to "
                    "their own department."
                ),
            )

    try:
        assignment = get_complaint_assignment(
            db=db,
            complaint_id=complaint.id,
        )

        if assignment is None:
            assignment = create_complaint_assignment(
                db=db,
                complaint_id=complaint.id,
                assigned_to_user_id=assigned_user.id,
                assigned_department_id=request.assigned_department_id,
                assignment_note=request.assignment_note,
                assigned_by_user_id=current_user.id,
            )

            create_assignment_history(
                db=db,
                complaint_id=complaint.id,
                previous_assigned_to_user_id=None,
                new_assigned_to_user_id=assignment.assigned_to_user_id,
                previous_department_id=None,
                new_department_id=assignment.assigned_department_id,
                changed_by_user_id=current_user.id,
                note=assignment.assignment_note,
            )

            action = "ASSIGN_COMPLAINT"
            details = (
                f"Assigned complaint to user {assigned_user.id} "
                f"({assigned_user.email})."
            )

        else:
            previous_assigned_user_id = assignment.assigned_to_user_id
            previous_department_id = assignment.assigned_department_id

            assignment = update_complaint_assignment(
                db=db,
                assignment=assignment,
                assigned_to_user_id=assigned_user.id,
                assigned_department_id=request.assigned_department_id,
                assignment_note=request.assignment_note,
                assigned_by_user_id=current_user.id,
            )

            create_assignment_history(
                db=db,
                complaint_id=complaint.id,
                previous_assigned_to_user_id=previous_assigned_user_id,
                new_assigned_to_user_id=assignment.assigned_to_user_id,
                previous_department_id=previous_department_id,
                new_department_id=assignment.assigned_department_id,
                changed_by_user_id=current_user.id,
                note=assignment.assignment_note,
            )

            action = "REASSIGN_COMPLAINT"
            details = (
                f"Reassigned complaint from user "
                f"{previous_assigned_user_id} to user "
                f"{assigned_user.id} ({assigned_user.email})."
            )

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action=action,
            entity_type="Complaint",
            entity_id=str(complaint.id),
            details=details,
        )

        return ComplaintAssignmentResponse(
            complaint_reference=complaint.complaint_reference,
            assigned_to_user_id=assignment.assigned_to_user_id,
            assigned_department_id=assignment.assigned_department_id,
            assignment_note=assignment.assignment_note,
            assigned_by_user_id=assignment.assigned_by_user_id,
            assigned_at=assignment.assigned_at,
            updated_at=assignment.updated_at,
        )

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(
            f"Assignment update failed for "
            f"{complaint_reference}: {error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complaint assignment could not be updated.",
        ) from error

@app.get(
    "/complaints/{complaint_reference}/escalation",
    response_model=ComplaintEscalationResponse,
)
def get_complaint_escalation_status(
    complaint_reference: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ComplaintEscalationResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

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

    escalation = get_complaint_escalation(
        db=db,
        complaint_id=complaint.id,
    )

    if escalation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No escalation record exists for this complaint.",
        )

    return ComplaintEscalationResponse(
        complaint_reference=complaint.complaint_reference,
        due_at=escalation.due_at,
        escalation_state=escalation.escalation_state,
        is_overdue=is_complaint_overdue(
            complaint=complaint,
            escalation=escalation,
        ),
        escalated_at=escalation.escalated_at,
        escalation_reason=escalation.escalation_reason,
        set_by_user_id=escalation.set_by_user_id,
        created_at=escalation.created_at,
        updated_at=escalation.updated_at,
    )


@app.put(
    "/complaints/{complaint_reference}/escalation",
    response_model=ComplaintEscalationResponse,
)
def set_complaint_escalation(
    complaint_reference: str,
    request: ComplaintEscalationUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Staff", "Admin")),
) -> ComplaintEscalationResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

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

    if complaint.status in {"Resolved", "Closed"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Escalation cannot be created or updated for a "
                "Resolved or Closed complaint."
            ),
        )

    updates = request.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one escalation field to update.",
        )

    escalation = get_complaint_escalation(
        db=db,
        complaint_id=complaint.id,
    )

    current_due_at = escalation.due_at if escalation else None
    current_state = (
        escalation.escalation_state
        if escalation is not None
        else "OnTrack"
    )
    current_reason = (
        escalation.escalation_reason
        if escalation is not None
        else None
    )

    due_at = updates.get("due_at", current_due_at)
    escalation_state = updates.get(
        "escalation_state",
        current_state,
    )
    escalation_reason = updates.get(
        "escalation_reason",
        current_reason,
    )

    if escalation_state not in ESCALATION_STATES:
        allowed_text = ", ".join(sorted(ESCALATION_STATES))

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid escalation_state. "
                f"Allowed values: {allowed_text}."
            ),
        )

    if escalation_state == "Escalated":
        if due_at is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "due_at is required when escalation_state is "
                    "Escalated."
                ),
            )

        if not escalation_reason:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "escalation_reason is required when "
                    "escalation_state is Escalated."
                ),
            )

    if "due_at" in updates and due_at is not None:
        if due_at <= datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="due_at must be in the future.",
            )

    if (
        "due_at" in updates
        and due_at is None
        and escalation_state == "Escalated"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "An Escalated complaint must have a due_at value."
            ),
        )

    try:
        if escalation is None:
            escalation = create_complaint_escalation(
                db=db,
                complaint_id=complaint.id,
                due_at=due_at,
                escalation_state=escalation_state,
                escalation_reason=escalation_reason,
                set_by_user_id=current_user.id,
            )
            action = "CREATE_COMPLAINT_ESCALATION"

        else:
            escalation = update_complaint_escalation(
                db=db,
                escalation=escalation,
                due_at=due_at,
                escalation_state=escalation_state,
                escalation_reason=escalation_reason,
                set_by_user_id=current_user.id,
            )
            action = "UPDATE_COMPLAINT_ESCALATION"

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action=action,
            entity_type="Complaint",
            entity_id=str(complaint.id),
            details=(
                f"Escalation state={escalation.escalation_state}; "
                f"due_at={escalation.due_at}; "
                f"is_overdue={is_complaint_overdue(complaint, escalation)}"
            ),
        )

        return ComplaintEscalationResponse(
            complaint_reference=complaint.complaint_reference,
            due_at=escalation.due_at,
            escalation_state=escalation.escalation_state,
            is_overdue=is_complaint_overdue(
                complaint=complaint,
                escalation=escalation,
            ),
            escalated_at=escalation.escalated_at,
            escalation_reason=escalation.escalation_reason,
            set_by_user_id=escalation.set_by_user_id,
            created_at=escalation.created_at,
            updated_at=escalation.updated_at,
        )

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(
            f"Escalation update failed for "
            f"{complaint_reference}: {error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complaint escalation could not be updated.",
        ) from error

@app.get(
    "/admin/complaints/unowned",
    response_model=UnownedComplaintListResponse,
)
def get_unowned_complaints(
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> UnownedComplaintListResponse:
    total, complaints = list_unowned_complaints(
        db=db,
        limit=limit,
        offset=offset,
    )

    return UnownedComplaintListResponse(
        total=total,
        complaints=[
            UnownedComplaintResponse(
                complaint_reference=complaint.complaint_reference,
                complaint_text=complaint.complaint_text,
                status=complaint.status,
                created_at=complaint.created_at,
            )
            for complaint in complaints
        ],
    )


@app.put(
    "/admin/complaints/{complaint_reference}/ownership",
    response_model=ComplaintOwnershipResponse,
)
def set_complaint_ownership(
    complaint_reference: str,
    request: ComplaintOwnershipUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> ComplaintOwnershipResponse:
    complaint = get_complaint_by_reference(
        db=db,
        complaint_reference=complaint_reference,
    )

    if complaint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint not found: {complaint_reference}",
        )

    submitted_by_user = get_user_by_id(
        db=db,
        user_id=request.submitted_by_user_id,
    )

    if submitted_by_user is None or not submitted_by_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected submitting user is unavailable.",
        )

    if submitted_by_user.role != "Student":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Complaint ownership can be assigned only to a Student.",
        )

    try:
        ownership = get_complaint_ownership(
            db=db,
            complaint_id=complaint.id,
        )

        if ownership is None:
            ownership = create_complaint_ownership(
                db=db,
                complaint_id=complaint.id,
                submitted_by_user_id=submitted_by_user.id,
            )

            action = "ASSIGN_COMPLAINT_OWNERSHIP"
            details = (
                f"Assigned complaint ownership to student "
                f"{submitted_by_user.id} ({submitted_by_user.email})."
            )

        else:
            previous_user_id = ownership.submitted_by_user_id

            if previous_user_id == submitted_by_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Complaint ownership is already assigned to this user.",
                )

            ownership = update_complaint_ownership(
                db=db,
                ownership=ownership,
                submitted_by_user_id=submitted_by_user.id,
            )

            action = "REASSIGN_COMPLAINT_OWNERSHIP"
            details = (
                f"Reassigned complaint ownership from user "
                f"{previous_user_id} to student "
                f"{submitted_by_user.id} ({submitted_by_user.email})."
            )

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action=action,
            entity_type="Complaint",
            entity_id=str(complaint.id),
            details=details,
        )

        return ComplaintOwnershipResponse(
            complaint_reference=complaint.complaint_reference,
            submitted_by_user_id=ownership.submitted_by_user_id,
            created_at=ownership.created_at,
        )

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(
            f"Complaint ownership update failed for "
            f"{complaint_reference}: {error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complaint ownership could not be updated.",
        ) from error

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

    if not can_access_complaint(
        db=db,
        complaint=complaint,
        current_user=current_user,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this complaint.",
        )
    existing_verification = get_complaint_verification(
        db=db,
        complaint_id=complaint.id,
    )

    if existing_verification is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Complaint impact has already been verified.",
        )

    if request.impact_verification_status not in IMPACT_VERIFICATION_STATUSES:
        allowed_text = ", ".join(
            sorted(IMPACT_VERIFICATION_STATUSES)
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid impact_verification_status. "
                f"Allowed values: {allowed_text}."
            ),
        )

    updates = request.model_dump(exclude_unset=True)

    if request.impact_verification_status == "Verified":
        if request.verified_affected_population is None:
            updates["verified_affected_population"] = (
                complaint.affected_population
            )

    elif request.impact_verification_status == "Adjusted":
        if request.verified_affected_population is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "verified_affected_population is required when "
                    "impact_verification_status is Adjusted."
                ),
            )

    elif request.impact_verification_status == "Rejected":
        updates["verified_affected_population"] = None

    try:
        verification = get_or_create_complaint_verification(
            db=db,
            complaint=complaint,
        )

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
            "verified_by_user_id": (
                updated_verification.verified_by_user_id
            ),
            "verified_at": updated_verification.verified_at,
        }

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(
            f"Impact verification failed for "
            f"{complaint_reference}: {error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complaint impact verification could not be updated.",
        ) from error


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

    if not can_access_complaint(
        db=db,
        complaint=complaint,
        current_user=current_user,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this complaint.",
        )

    updates = request.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one ML feedback field to update.",
        )

    if (
        "final_priority" in updates
        and updates["final_priority"] is not None
        and updates["final_priority"] not in FINAL_PRIORITIES
    ):
        allowed_text = ", ".join(sorted(FINAL_PRIORITIES))

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid final_priority. "
                f"Allowed values: {allowed_text}."
            ),
        )

    if (
        "duplicate_decision" in updates
        and updates["duplicate_decision"] is not None
        and updates["duplicate_decision"] not in DUPLICATE_DECISIONS
    ):
        allowed_text = ", ".join(sorted(DUPLICATE_DECISIONS))

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid duplicate_decision. "
                f"Allowed values: {allowed_text}."
            ),
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

    if (
        updates.get("training_eligible") is True
        and current_user.role != "Admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Only Admin users can mark feedback records "
                "as training eligible."
            ),
        )

    if updates.get("training_eligible") is True:
        if complaint.status not in {"Resolved", "Closed"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Complaint must be Resolved or Closed before it can "
                    "be marked training eligible."
                ),
            )

    try:
        feedback = get_or_create_ml_feedback_record(
            db=db,
            complaint=complaint,
        )

        if updates.get("training_eligible") is True:
            required_fields = [
                "final_category",
                "final_department_id",
                "final_priority",
                "actual_resolution_hours",
                "duplicate_decision",
            ]

            combined_values = {
                field_name: updates.get(
                    field_name,
                    getattr(feedback, field_name),
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
                        "Cannot mark training eligible. "
                        f"Missing verified fields: {missing_text}"
                    ),
                )

            if combined_values["duplicate_decision"] == "NotReviewed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Cannot mark training eligible while "
                        "duplicate_decision is NotReviewed."
                    ),
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
            actual_resolution_hours=(
                updated_feedback.actual_resolution_hours
            ),
            duplicate_decision=updated_feedback.duplicate_decision,
            training_eligible=updated_feedback.training_eligible,
            exclusion_reason=updated_feedback.exclusion_reason,
            reviewed_by_user_id=updated_feedback.reviewed_by_user_id,
            reviewed_at=updated_feedback.reviewed_at,
        )

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(
            f"ML feedback update failed for "
            f"{complaint_reference}: {error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ML feedback could not be updated.",
        ) from error

@app.get(
    "/admin/audit-logs",
    response_model=AuditLogListResponse,
)
def get_audit_logs(
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    action: str | None = Query(default=None),
    actor_user_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> AuditLogListResponse:
    total, logs = list_audit_logs(
        db=db,
        entity_type=entity_type.strip() if entity_type else None,
        entity_id=entity_id.strip() if entity_id else None,
        action=action.strip() if action else None,
        actor_user_id=actor_user_id,
        limit=limit,
        offset=offset,
    )

    return AuditLogListResponse(
        total=total,
        logs=[
            AuditLogResponse(
                id=log.id,
                actor_user_id=log.actor_user_id,
                action=log.action,
                entity_type=log.entity_type,
                entity_id=log.entity_id,
                details=log.details,
                created_at=log.created_at,
            )
            for log in logs
        ],
    )


@app.post(
    "/admin/ml/training-export",
    response_model=MLTrainingExportResponse,
)
def export_ml_training_dataset(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> MLTrainingExportResponse:
    try:
        records = list_training_eligible_records(db=db)

        result = export_training_feedback_dataset(records)

        create_audit_log(
            db=db,
            actor_user_id=current_user.id,
            action="EXPORT_ML_TRAINING_DATASET",
            entity_type="MLTrainingDataset",
            entity_id=result["dataset_version"],
            details=(
                f"Exported {result['record_count']} eligible records; "
                f"sha256={result['sha256']}"
            ),
        )

        return MLTrainingExportResponse(**result)

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print(f"ML training dataset export failed: {error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ML training dataset export could not be completed.",
        ) from error


@app.get(
    "/admin/ml/monitoring",
    response_model=MLMonitoringResponse,
)
def get_ml_monitoring(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Admin")),
) -> MLMonitoringResponse:
    return MLMonitoringResponse(
        **get_ml_monitoring_summary(db=db)
    )


@app.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
def dashboard_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DashboardSummaryResponse:
    submitted_by_user_id = (
        current_user.id
        if current_user.role == "Student"
        else None
    )

    return DashboardSummaryResponse(
        **get_dashboard_summary(
            db=db,
            submitted_by_user_id=submitted_by_user_id,
        )
    )