"""
Model Registry & Version Management Routes.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from src.modeling.registry import ModelArtifactRegistry
from api.schemas.responses import ModelDetailsResponse
from src.utils.logger import setup_logger

logger = setup_logger("api.routes.models")
router = APIRouter(prefix="/models", tags=["Model Registry"])

registry = ModelArtifactRegistry()


@router.get("", response_model=List[Dict[str, Any]])
def list_models():
    """List all trained model targets and available versions."""
    targets = ["PM10", "PM2.5"]
    output = []
    for target in targets:
        versions = registry.list_versions(target)
        output.append({"target": target, "versions": versions})
    return output


@router.get("/{model_id}", response_model=ModelDetailsResponse)
def get_model_details(model_id: str):
    """
    Get model artifact metadata by identifier format 'TARGET_VERSION' (e.g. PM10_v1.0.0 or PM10:v1.0.0).
    """
    separator = ":" if ":" in model_id else "_"
    parts = model_id.split(separator, 1)

    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid model_id format. Expected 'TARGET_VERSION' e.g. PM10_v1.0.0")

    target, version = parts[0], parts[1]

    try:
        artifact = registry.load_artifact(target=target, version=version)
        return ModelDetailsResponse(
            target=target,
            version=version,
            metadata=artifact["metadata"],
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model artifact '{model_id}' not found: {e}")
