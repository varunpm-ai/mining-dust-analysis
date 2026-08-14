"""
Dataset Upload & Management Routes.
"""

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas.responses import DatasetUploadResponse
from src.utils.hashing import calculate_file_hash
from src.utils.exceptions import DataIngestionError
from src.utils.logger import setup_logger

logger = setup_logger("api.routes.datasets")
router = APIRouter(prefix="/datasets", tags=["Datasets"])

UPLOAD_DIR = Path("artifacts/datasets/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload an Excel (.xlsx, .xls) or CSV (.csv) dataset for processing.
    """
    filename = file.filename
    if not filename:
        raise DataIngestionError(message="No file uploaded.", user_message="Please select a valid file to upload.")

    ext = Path(filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise DataIngestionError(
            message=f"Unsupported file format '{ext}'.",
            user_message="Only CSV (.csv) and Excel (.xlsx, .xls) files are supported.",
        )

    destination = UPLOAD_DIR / filename
    content = await file.read()
    
    if len(content) == 0:
        raise DataIngestionError(message="Empty file uploaded.", user_message="Uploaded file is empty.")

    with open(destination, "wb") as f:
        f.write(content)

    file_hash = calculate_file_hash(destination)
    logger.info(f"Uploaded dataset '{filename}' ({len(content)} bytes, hash: {file_hash[:8]})")

    return DatasetUploadResponse(
        message="Dataset uploaded successfully.",
        filename=filename,
        filepath=str(destination.resolve()),
        dataset_hash=file_hash,
        file_size_bytes=len(content),
    )
