# CampusResolve-AI

> A multilingual, explainable complaint intelligence system for campus environments.

CampusResolve-AI is an end-to-end Machine Learning and NLP research prototype that analyzes campus complaints written in English, Tamil, or Tamil-English (Tanglish). It predicts complaint category, routes the complaint to the appropriate department, detects potential duplicate complaints, forecasts priority, estimates resolution time, and provides transparent explanations for its predictions.

## Key Features

- Multilingual complaint intake for English, Tamil, and Tanglish
- Complaint category classification across 10 operational categories
- Automated department-routing recommendation
- Semantic duplicate detection using Sentence Transformers and FAISS
- Priority forecasting: Low, Medium, High, and Critical
- Resolution-time prediction with uncertainty interval
- Explainable AI using feature contributions and SHAP-style model explanations
- FastAPI backend with automatically generated Swagger API documentation
- Streamlit student complaint portal and staff operations dashboard
- SQLite complaint storage with staff queue, status updates, and notes
- Automated testing with Pytest
- Docker Compose configuration for reproducible containerized deployment

## Architecture

```text
┌──────────────────────────────┐
│       Streamlit Frontend     │
│ Student Portal + Staff Panel │
└──────────────┬───────────────┘
               │ HTTP / JSON
               ▼
┌──────────────────────────────┐
│        FastAPI Backend       │
│ Validation + Routing + APIs  │
└──────────────┬───────────────┘
               │
     ┌─────────┼─────────────────────────────────────────┐
     ▼         ▼                  ▼                        ▼
┌──────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│ Category │ │ Priority     │ │ Resolution   │ │ Duplicate      │
│ Model    │ │ Model        │ │ Time Model   │ │ Detection      │
│ TF-IDF + │ │ Structured + │ │ Random       │ │ Sentence       │
│ LogReg   │ │ LogReg       │ │ Forest       │ │ Transformers + │
└──────────┘ └──────────────┘ └──────────────┘ │ FAISS          │
                                                └────────────────┘
               │
               ▼
┌──────────────────────────────┐
│      SQLite / PostgreSQL      │
│ Complaints + Status + Notes   │
└──────────────────────────────┘
```

## Complaint Categories

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

## Technology Stack

| Layer | Tools |
|---|---|
| Programming language | Python 3.12 |
| Machine learning | Scikit-learn, Pandas, NumPy |
| NLP and embeddings | Sentence Transformers |
| Vector retrieval | FAISS |
| Explainability | SHAP, model feature contributions |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Database | SQLite with SQLAlchemy; PostgreSQL-ready |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| Version control | Git and GitHub |

## Project Structure

```text
CampusResolve-AI/
├── apps/
│   ├── api/                        # FastAPI backend
│   │   ├── main.py
│   │   ├── schemas/
│   │   └── services/
│   └── dashboard/
│       └── streamlit_app.py        # Streamlit user and staff dashboard
├── artifacts/                      # Saved trained models and FAISS index
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/                       # Data generation and preparation
│   ├── database/                   # SQLAlchemy models and repositories
│   ├── explainability/             # Prediction explanations
│   ├── features/                   # Risk feature engineering
│   ├── models/                     # Training pipelines
│   └── retrieval/                  # FAISS duplicate detection
├── tests/                          # Automated tests
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CampusResolve-AI.git
cd CampusResolve-AI
```

### 2. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Generate and prepare the dataset

```powershell
python -m src.data.generate_dataset
python -m src.data.prepare_dataset
```

### 5. Train ML models

```powershell
python -m src.models.train_category_classifier
python -m src.retrieval.build_duplicate_index
python -m src.models.train_priority_classifier
python -m src.models.train_resolution_regressor
python -m src.models.train_priority_classifier_improved
```

### 6. Initialize database

```powershell
python -m src.database.init_db
```

### 7. Run locally

Start the FastAPI backend:

```powershell
uvicorn apps.api.main:app --reload
```

In a separate terminal, activate the virtual environment and start Streamlit:

```powershell
streamlit run apps.dashboard.streamlit_app
```

Open:

- FastAPI API documentation: `http://127.0.0.1:8000/docs`
- Streamlit application: `http://localhost:8501`

## Testing

Run all tests:

```powershell
python -m pytest
```

Current automated test coverage includes:

- FastAPI root and health endpoints
- API request validation
- Database complaint-reference generation
- Complaint queue filtering
- Status and staff-note updates
- Dashboard summary aggregation
- Mocked prediction and complaint-creation endpoints

## ML Evaluation Summary

| Module | Main evaluation result |
|---|---|
| Category classification | Accuracy: 1.0000; Macro F1: 1.0000 |
| Priority forecasting baseline | Accuracy: 0.3527; Macro F1: 0.3469 |
| Improved priority forecasting | Accuracy: 0.3565; Macro F1: 0.3561 |
| Resolution-time prediction | Test MAE: 10.8971 hours; Test RMSE: 17.8884 hours; Test \(R^2\): 0.6122 |
| Duplicate detection historical evaluation | Threshold: 0.65; Precision: 0.2325; Recall: 0.9944; F1: 0.3768; MRR: 0.4091 |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API root and documentation link |
| `GET` | `/health` | Service/model/database health information |
| `POST` | `/predict` | Analyze a complaint without storing it |
| `POST` | `/complaints` | Analyze and store a complaint |
| `GET` | `/complaints` | List/filter stored complaints |
| `GET` | `/complaints/{complaint_reference}` | Get one stored complaint |
| `PATCH` | `/complaints/{complaint_reference}` | Update status and staff notes |
| `GET` | `/dashboard/summary` | Operational dashboard summary |

## Dataset and Ethics

The current dataset is synthetic and designed to validate the complete ML/NLP system workflow. It does not contain real student names, roll numbers, phone numbers, email addresses, or other personally identifiable information.

Important limitations:

- Synthetic, template-generated data can inflate category-classification scores.
- Synthetic priority labels include random components, which limits forecasting accuracy.
- Duplicate detection currently encounters semantically repetitive synthetic complaint patterns.
- This application is decision support only. Campus staff must make all final decisions about routing, urgency, escalation, duplicate merging, and complaint closure.

## Docker Deployment

Docker configuration files are included:

```powershell
docker compose up --build
```

After successful startup:

- FastAPI: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

## Future Improvements

- Fine-tune multilingual embeddings using real anonymized campus complaint pairs
- Add authentication and role-based access control for students and staff
- Move production data from SQLite to PostgreSQL
- Add real SLA policies by category and department
- Add staff feedback loop for duplicate labels and priority correction
- Add notification channels such as email or SMS
- Add monitoring, audit logs, and fairness analysis
- Deploy to cloud infrastructure

## Author

**Premkumar S**

M.Tech Data Science  
Individual ML/NLP Research Project

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.