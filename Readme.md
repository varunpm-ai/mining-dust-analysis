# Mine Dust Analysis — Version 2.0 Production MLOps Platform

Production-grade MLOps platform for automated mine dust data ingestion, statistical analysis, multi-model regression training, evaluation, model selection, versioning, and executive reporting.

---

## 🏗️ Target System Architecture

```text
Client (Web Dashboard)
       │
       ▼
Streamlit UI (`app/streamlit_app.py`)
       │
       ▼ REST API Calls
FastAPI Backend (`api/main.py`)
       │
       ▼ Engine Execution
ML Pipeline Engine (`src/`)
  ├── Ingestion & Normalization (`src/ingestion/`)
  ├── Schema & Preprocessing (`src/preprocessing/`)
  ├── Statistical Analysis (`src/statistics/`)
  ├── ML Training & Selection (`src/modeling/`)
  └── Reporting & Visualization (`src/reporting/`, `src/analysis/`)
       │
       ▼ Versioned Storage
Artifact Storage (`artifacts/`)
       │
       ▼ CI/CD Automation
Jenkins Pipeline (`Jenkinsfile`)
```

---

## 📁 Repository Structure

```text
model_training/
├── app/                  # Streamlit Dashboard UI
├── api/                  # FastAPI Web Backend Services
│   ├── routes/           # Endpoint handlers
│   ├── schemas/          # Pydantic request/response schemas
│   └── services/        # Orchestration services
├── src/                  # Core Production ML Engine
│   ├── ingestion/        # Excel/CSV loader & header normalizers
│   ├── preprocessing/    # Cleaning, imputation, & transformations
│   ├── analysis/         # Exploratory data analysis helpers
│   ├── statistics/       # Statistical tests, correlations, & MLR
│   ├── modeling/         # Model definitions, trainer, evaluation, & selection
│   ├── reporting/        # Automated executive report & chart generation
│   └── utils/            # Logging, exceptions, hashing, configuration
├── configs/              # System & schema YAML configurations
├── artifacts/            # Model registries, reports, metrics, datasets
├── tests/                # Unit & Integration test suite
├── notebooks/            # Research prototypes (Version 1.0 - Read Only)
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-container service composition
└── Jenkinsfile           # CI/CD automation pipeline
```

---

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   ```

3. **Run Unit Tests**:
   ```bash
   pytest
   ```
