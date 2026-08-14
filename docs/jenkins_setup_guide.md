# Jenkins CI/CD Setup Guide for Mine Dust MLOps Platform

## Overview
This document describes the exact Jenkins configuration required to run the automated Machine Learning training, statistical analysis, model evaluation, semantic versioning, and report generation pipeline.

---

## 1. Prerequisites
- **Jenkins Server**: Version 2.400+ with Pipeline and HTML Publisher plugins installed.
- **Node / Agent Requirements**:
  - Python 3.11+ installed and available on `PATH`.
  - Git client installed.
  - Required Python packages installed (`requirements.txt`).
- **Disk Storage**: Sufficient space in `artifacts/models/` for immutable versioned model artifacts.

---

## 2. Jenkins Pipeline Configuration Steps

### Step 1: Create a New Pipeline Job
1. Log in to Jenkins Dashboard.
2. Click **New Item**.
3. Enter item name: `mine-dust-ml-training-pipeline`.
4. Select **Pipeline** and click **OK**.

### Step 2: Configure Job Parameters
Check **This project is parameterized** and add the following parameters:

1. **Choice Parameter**:
   - Name: `TARGET`
   - Choices: `PM10`, `PM2.5`
   - Description: Target air quality metric for training and evaluation.

2. **String Parameter**:
   - Name: `DATASET_PATH`
   - Default Value: `artifacts/datasets/uploads/jenkins_input.csv`
   - Description: File path to the operational mining dataset.

3. **Choice Parameter**:
   - Name: `SELECTION_METRIC`
   - Choices: `RMSE`, `MAE`, `R2`
   - Description: Primary metric for selecting the winning model.

4. **Choice Parameter**:
   - Name: `TRAINING_MODE`
   - Choices: `full`, `fast`, `cross_validation`
   - Description: Execution speed and cross-validation depth.

---

### Step 3: Configure Pipeline Definition
1. Under **Pipeline**, set **Definition** to **Pipeline script from SCM**.
2. Set **SCM** to **Git**.
3. Repository URL: `<YOUR_GIT_REPOSITORY_URL>`
4. Credentials: Select Jenkins credentials for repository access.
5. Branch Specifier: `*/main` or `*/master`.
6. Script Path: `Jenkinsfile`.

---

## 3. Pipeline Lifecycle & Recorded Build Metrics

The pipeline executes the following 10 lifecycle stages in order:

```
[1. CHECKOUT] -> [2. VALIDATE] -> [3. PREPROCESS] -> [4. STATISTICAL ANALYSIS] -> [5. TRAIN MODELS]
                                                                                          |
[10. ARCHIVE ARTIFACTS] <- [9. GENERATE REPORT] <- [8. VERSION MODEL] <- [7. SELECT BEST MODEL] <- [6. EVALUATE]
```

### Recorded Build Metrics (`artifacts/jenkins_build_summary.json`):
Each Jenkins build records the following metadata in the model artifact:
- `git_commit`: Short & long Git commit SHA
- `jenkins_build_number`: Jenkins build ID (`BUILD_NUMBER`)
- `dataset_hash`: SHA256 cryptographic hash of raw input dataset
- `model_version`: Assigned immutable semantic version (e.g. `v1.0.12`)
- `selected_model`: Winning algorithm name (e.g. `Support Vector Regressor (SVR)`)
- `MAE`: Mean Absolute Error on out-of-fold CV
- `RMSE`: Root Mean Squared Error on out-of-fold CV
- `R2`: Coefficient of Determination ($R^2$)

---

## 4. Immutable Model Version Preservation

- Model artifacts are saved in `artifacts/models/<TARGET>/v1.0.X/`.
- Previous model versions are **never overwritten**.
- A new semantic version directory is automatically generated for every training run.
- Old model versions remain accessible via the REST API (`/models/{model_id}`) and model registry.

---

## 5. Security & Secret Management

- No passwords, API keys, or database credentials are stored in source code.
- Credentials and production environment overrides are managed via Jenkins Credentials Manager or `.env.production`.
