"""
Client-Facing Streamlit Application for Mine Dust Analysis Platform.

Communicates with FastAPI Backend API to provide a non-technical UI for:
1. CSV/XLSX dataset upload
2. Pipeline execution with stage-by-stage progress monitoring (VALIDATING -> PREPROCESSING -> ANALYZING -> TRAINING -> EVALUATING -> SELECTING -> REPORTING -> COMPLETED / FAILED)
3. Quality audit & statistical analysis visualization
4. ML model comparison & winning model identification
5. Downloading reports, processed datasets, model artifacts, and metrics
"""

import os
import json
import time
import requests
import streamlit as st
import pandas as pd

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="Mine Dust Analysis — Production MLOps",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 25px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        height: 45px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Verify backend API connectivity."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/coal.png", width=64)
    st.title("Control Panel")

    api_healthy = check_api_health()
    if api_healthy:
        st.success("🟢 API Service Connected")
    else:
        st.error("🔴 API Service Disconnected")
        st.info("Ensure the FastAPI backend is running on `http://127.0.0.1:8000`.")

    st.divider()
    st.subheader("Analysis Parameters")
    target_col = st.selectbox("Target Variable", ["PM10", "PM2.5"], index=0)
    cv_folds = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5)
    primary_metric = st.selectbox("Primary Model Selection Metric", ["RMSE", "MAE", "R2"], index=0)

    st.divider()
    st.caption("Mine Dust Analysis Production MLOps Platform v2.0")


# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">⛏️ Mine Dust Analysis & Predictive Modeling System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated data validation, statistical regression, and machine learning pipeline for environmental dust monitoring.</div>', unsafe_allow_html=True)

# Initialize Session State
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "dataset_hash" not in st.session_state:
    st.session_state.dataset_hash = None
if "current_job_id" not in st.session_state:
    st.session_state.current_job_id = None
if "job_results" not in st.session_state:
    st.session_state.job_results = None


# STEP 1: UPLOAD DATASET
st.header("1. Upload Operational Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel file (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if st.button("📤 Upload File to Platform"):
        with st.spinner("Uploading and calculating SHA-256 hash..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{API_BASE_URL}/datasets/upload", files=files, timeout=30)
                
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.uploaded_filename = data["filename"]
                    st.session_state.dataset_hash = data["dataset_hash"]
                    st.success(f"File '{data['filename']}' uploaded successfully! (SHA-256: `{data['dataset_hash'][:12]}...`)")
                else:
                    err = res.json()
                    st.error(f"Upload failed: {err.get('user_message', 'Unknown error')}")
            except Exception as e:
                st.error("Unable to connect to backend server. Please verify backend service.")

if st.session_state.uploaded_filename:
    st.info(f"📁 Active Dataset: **{st.session_state.uploaded_filename}**")


# STEP 2: RUN ANALYSIS & STAGE-BY-STAGE PROGRESS
st.header("2. Run Automated Analysis Pipeline")
if st.button("🚀 Start Automated Analysis"):
    if not st.session_state.uploaded_filename:
        st.warning("Please upload a dataset file first before starting analysis.")
    elif not api_healthy:
        st.error("Backend API is disconnected. Cannot start analysis.")
    else:
        progress_bar = st.progress(0, text="Initializing Pipeline...")
        status_box = st.empty()

        try:
            payload = {
                "dataset_filename": st.session_state.uploaded_filename,
                "target_column": target_col,
                "cv_folds": cv_folds,
                "primary_metric": primary_metric,
            }
            res = requests.post(f"{API_BASE_URL}/analysis/start", json=payload, timeout=180)

            if res.status_code == 200:
                job_info = res.json()
                job_id = job_info["job_id"]
                st.session_state.current_job_id = job_id
                
                stage = job_info.get("status", "COMPLETED")
                progress_bar.progress(100, text="STAGE: COMPLETED - Pipeline Finished!")
                status_box.success(f"✅ Pipeline job `{job_id}` finished successfully!")

                # Retrieve Results
                results_res = requests.get(f"{API_BASE_URL}/analysis/{job_id}/results", timeout=10)
                if results_res.status_code == 200:
                    st.session_state.job_results = results_res.json()
                else:
                    st.error("Failed to retrieve analysis results.")
            else:
                err = res.json()
                progress_bar.progress(100, text="STAGE: FAILED")
                status_box.error(f"❌ Analysis failed: {err.get('user_message', 'Pipeline failure')}")
        except Exception as e:
            progress_bar.progress(100, text="STAGE: FAILED")
            status_box.error("Network communication failure with FastAPI backend service.")


# STEP 3: DISPLAY RESULTS & DASHBOARD
if st.session_state.job_results:
    results = st.session_state.job_results
    job_id = st.session_state.current_job_id

    st.header("3. Executive Analysis Results")

    # Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    q_rep = results.get("quality_report", {})
    col1.metric("Ingested Records", q_rep.get("initial_rows", "N/A"))
    col2.metric("Cleaned Records", q_rep.get("final_rows", "N/A"))
    col3.metric("Winning Model", results.get("winning_model_name", "N/A"))
    winning_metrics = results.get("winning_metrics", {})
    col4.metric(f"Best {primary_metric}", f"{winning_metrics.get(primary_metric, 0.0):.4f}")

    # Tabs for Detailed Views
    tab1, tab2, tab3 = st.tabs(["📊 ML Model Comparison", "📋 Ingestion Audit", "📥 Downloads & Artifact Exports"])

    with tab1:
        st.subheader("Regression Algorithm Performance Comparison")
        st.write("Out-of-fold validation metrics across all 7 trained regression models (ranked by performance):")
        comp_list = results.get("comparison_table", [])
        if comp_list:
            df_comp = pd.DataFrame(comp_list)
            st.dataframe(df_comp, use_container_width=True)

    with tab2:
        st.subheader("Data Quality & Cleaning Summary")
        st.json(q_rep)

    with tab3:
        st.subheader("Download Pipeline Artifacts & Reports")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        report_url = f"{API_BASE_URL}/analysis/{job_id}/report"
        d_col1.markdown(f"[📄 Download HTML Report]({report_url})")

        dataset_url = f"{API_BASE_URL}/analysis/{job_id}/dataset"
        d_col2.markdown(f"[📊 Download Cleaned Dataset]({dataset_url})")

        model_url = f"{API_BASE_URL}/analysis/{job_id}/model"
        d_col3.markdown(f"[⚙️ View Model Details]({model_url})")

        metrics_json_str = json.dumps(results, indent=4)
        d_col4.download_button(
            label="💾 Download Metrics JSON",
            data=metrics_json_str,
            file_name=f"metrics_{job_id}.json",
            mime="application/json",
        )
