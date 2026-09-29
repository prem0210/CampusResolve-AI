# CampusResolve-AI

CampusResolve-AI is a multilingual, AI-assisted campus complaint-management application. It helps campus users submit and track complaints written in English, Tamil, or Tamil-English (Tanglish), while providing Staff and Admin users with secure operational workflows.

The application analyzes complaint text to recommend a category and responsible department, forecast priority, estimate resolution time, and flag potentially similar historic complaints. AI recommendations support operational decisions; authorized campus staff make the final decisions.

## Features

### Complaint intelligence

- Multilingual complaint intake for English, Tamil, and Tamil-English.
- Complaint-category prediction across campus operational categories.
- Department-routing recommendation.
- Priority forecasting: Low, Medium, High, and Critical.
- Resolution-time estimation with an uncertainty interval.
- Potential duplicate detection using semantic similarity, Sentence Transformers, and FAISS.
- Explainable recommendation output.

### Secure workflow management

- JWT-based authentication.
- Student, Staff, and Admin role-based access control.
- Authenticated complaint submission and complaint tracking.
- Complaint ownership and access control.
- Staff complaint queue with filters, status updates, and staff notes.
- Assignment, escalation, and status-history visibility.
- One-time Staff/Admin impact verification of reported affected population.
- Audit logging for protected operational actions.
- SQLite persistence using SQLAlchemy.

### Application interfaces

- FastAPI REST backend with Swagger documentation.
- Streamlit dashboard for Student and Staff workflows.
- Automated backend regression tests.
- Docker Compose configuration for local containerized deployment.

## Architecture

```text
Student / Staff / Admin
          |
          v
Streamlit Dashboard
Submit | Track | Staff Operations
          |
          v
FastAPI Backend
Authentication | Complaint APIs | Workflow APIs
          |
     +----+--------------------+
     |                         |
     v                         v
Security Layer            AI Intelligence
JWT | RBAC | Audit        Category | Priority
Ownership                 Resolution Time | Duplicates
     |                         |
     +------------+------------+
                  |
                  v
SQLite Database with SQLAlchemy
Users | Complaints | Assignments | Verifications | Audit Logs
```

## Technology stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Database | SQLite, SQLAlchemy |
| Authentication | JWT, `python-jose`, password hashing with `pwdlib` |
| Machine learning | Scikit-learn, Pandas, NumPy |
| Multilingual NLP | Sentence Transformers |
| Similarity retrieval | FAISS |
| Explainability | SHAP and model feature contributions |
| Testing | Pytest |
| Containers | Docker, Docker Compose |
| Version control | Git and GitHub |

## ML components

| Component | Purpose |
|---|---|
| Category classifier | Predicts the operational category for a complaint. |
| Priority classifier | Recommends Low, Medium, High, or Critical priority. |
| Resolution-time regressor | Estimates expected complaint-resolution time. |
| Sentence Transformer + FAISS index | Retrieves semantically similar historic complaints as potential duplicates. |
| Department-routing logic | Recommends the responsible department from the complaint analysis. |

## Complaint categories

| Category | Default department |
|---|---|
| Electrical and Power | Electrical Maintenance |
| Water and Plumbing | Plumbing and Civil Maintenance |
| Hostel and Accommodation | Hostel Administration |
| Wi-Fi and IT Services | IT Support and Network Cell |
| Classroom and Laboratory | Academic Infrastructure |
| Cleanliness and Waste | Housekeeping and Sanitation |
| Safety and Security | Security Office |
| Transport and Parking | Transport Cell |
| Administration and Documents | Administrative Office |
| Food and Canteen | Canteen Committee |

## Project structure

```text
CampusResolve-AI/
â”œâ”€â”€ apps/
â”‚   â”œâ”€â”€ api/                         # FastAPI backend
â”‚   â”‚   â”œâ”€â”€ core/                    # Security, dependencies, workflow rules
â”‚   â”‚   â”œâ”€â”€ schemas/                 # Pydantic API schemas
â”‚   â”‚   â””â”€â”€ main.py                  # API entry point
â”‚   â””â”€â”€ dashboard/
â”‚       â””â”€â”€ streamlit_app.py         # Streamlit Student and Staff dashboard
â”œâ”€â”€ artifacts/                       # Saved ML models and FAISS index
â”œâ”€â”€ data/                            # Raw and processed datasets
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/                        # Dataset generation and preparation
â”‚   â”œâ”€â”€ database/                    # SQLAlchemy models and repositories
â”‚   â”œâ”€â”€ explainability/              # Explanation utilities
â”‚   â”œâ”€â”€ features/                    # Feature engineering
â”‚   â”œâ”€â”€ models/                      # Training pipelines
â”‚   â””â”€â”€ retrieval/                   # Duplicate-detection utilities
â”œâ”€â”€ tests/                           # Automated tests
â”œâ”€â”€ Dockerfile.api
â”œâ”€â”€ Dockerfile.streamlit
â”œâ”€â”€ docker-compose.yml
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

## Prerequisites

- Python 3.12
- Git
- PowerShell on Windows, or an equivalent terminal
- Optional: Docker Desktop / Docker Engine with Docker Compose

## Local setup

### 1. Clone the repository

```powershell
git clone [https://github.com/prem0210/CampusResolve-AI.git](https://github.com/prem0210/CampusResolve-AI.git)
Set-Location CampusResolve-AI
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use a session-only policy change:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Create a local `.env` file from the template:

```powershell
Copy-Item .env.example .env
```

Generate a secret locally:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Paste the generated value into the local `.env` file:

```dotenv
JWT_SECRET_KEY=<your-private-random-secret>
```

Do not commit `.env`, database files, access tokens, or real credentials.

The default local database setting is:

```dotenv
DATABASE_URL=sqlite:///./campusresolve.db
```

### 5. Initialize the database

```powershell
python -m src.database.init_db
```

### 6. Start the FastAPI backend

```powershell
uvicorn apps.api.main:app --reload
```

Open FastAPI Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

### 7. Start the Streamlit dashboard

Open a second terminal in the project directory, activate the virtual environment, and run:

```powershell
streamlit run apps\dashboard\streamlit_app.py
```

Open the dashboard at:

```text
http://localhost:8501
```

## User workflows

### Student

1. Sign in through the dashboard.
2. Submit a complaint with location and impact information.
3. View AI-assisted category, department, priority, resolution-time, and duplicate recommendations.
4. Track accessible complaints using the complaint reference.
5. Review status updates and staff notes.

### Staff

1. Sign in with a Staff account.
2. Open the Staff operations workspace.
3. Filter and review the complaint queue.
4. Update complaint status and add staff notes.
5. Review duplicate flags and AI recommendations.
6. Verify the reported affected population as Verified, Adjusted, or Rejected.
7. Review assignment, escalation, and status-history details.

### Admin

Admin users have Staff capabilities plus access to authorized administrative functions such as audit and operational management endpoints, depending on the configured deployment workflow.

## Testing

Run the automated test suite:

```powershell
pytest -q
```

Current validation result:

```text
88 passed
```

The suite covers API smoke checks, authentication schemas, complaint workflows, prediction APIs, repository operations, and request validation.

## Key API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/login` | Authenticate a user and return a JWT access token. |
| `GET` | `/auth/me` | Return the authenticated user profile. |
| `GET` | `/health` | Service, model, and database health information. |
| `POST` | `/predict` | Analyze a complaint without storing it. |
| `POST` | `/complaints` | Analyze and store an authenticated complaint. |
| `GET` | `/complaints` | List/filter accessible complaints. |
| `GET` | `/complaints/{complaint_reference}` | Retrieve an accessible complaint. |
| `PATCH` | `/complaints/{complaint_reference}` | Update complaint status and staff notes. |
| `GET` | `/complaints/{complaint_reference}/timeline` | Retrieve assignment, impact, escalation, and status history. |
| `PATCH` | `/complaints/{complaint_reference}/impact-verification` | Verify reported impact; Staff/Admin only. |
| `GET` | `/dashboard/summary` | Retrieve operational dashboard metrics. |

Use the Swagger UI for the complete API reference.

## Docker deployment

Docker Compose starts the API and Streamlit dashboard:

```powershell
docker compose up --build
```

After startup:

```text
FastAPI: http://localhost:8000/docs
Streamlit: http://localhost:8501
```

Docker uses a named volume for SQLite persistence. Before relying on Docker for a demo or submission, verify the container build in your local environment.

## Data and security notes

- The current dataset is synthetic and does not contain real student personal data.
- AI outputs are decision-support recommendations; authorized Staff make final operational decisions.
- JWT tokens are stored in Streamlit session state during the active session.
- The application protects authenticated workflows using bearer-token authorization and role checks.
- `.env`, database files, virtual environments, backups, and Streamlit secrets are excluded from Git.
- Do not enter passwords, tokens, identity documents, or other sensitive personal data in complaint text.

## Known limitations

- Current ML evaluation is based on synthetic data and may not reflect performance on live campus data.
- Potential duplicate detection flags candidates for Staff review; it does not automatically merge complaints.
- SQLite is suitable for local development and demos; a production deployment should use a managed database and operational monitoring.
- The dashboard is intended for local or controlled deployment; production use requires additional security, privacy, backup, and monitoring controls.

## Author

**Premkumar S**


M.Tech Data Science
Rajalakshmi Engineering College

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
