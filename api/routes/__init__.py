"""
API Routers Package.
"""

from api.routes.health import router as health_router
from api.routes.datasets import router as datasets_router
from api.routes.analysis import router as analysis_router
from api.routes.models import router as models_router

__all__ = [
    "health_router",
    "datasets_router",
    "analysis_router",
    "models_router",
]
