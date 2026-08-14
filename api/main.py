"""
FastAPI Application Entry Point with Centralized Exception Handling and Middleware.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health_router, datasets_router, analysis_router, models_router
from api.schemas.responses import ErrorResponse
from src.utils.exceptions import MineDustException
from src.utils.logger import setup_logger

logger = setup_logger("api.main")

app = FastAPI(
    title="Mine Dust Analysis Production MLOps API",
    description="FastAPI Backend for automated mine dust dataset validation, statistical analysis, ML regression modeling, and reporting.",
    version="2.0.0",
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health_router)
app.include_router(datasets_router)
app.include_router(analysis_router)
app.include_router(models_router)


# Centralized Custom Exception Handlers (Never expose Python tracebacks to clients)
@app.exception_handler(MineDustException)
async def minedust_exception_handler(request: Request, exc: MineDustException):
    """Handle domain-specific custom exceptions with user-safe error messages."""
    logger.error(f"Domain Exception [{exc.code}] on {request.url}: {exc.message}", exc_info=True)
    status_code = 400
    if exc.code in ["JOB_NOT_FOUND", "DATASET_NOT_FOUND", "MODEL_NOT_FOUND"]:
        status_code = 404

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            code=exc.code,
            user_message=exc.user_message,
            details=exc.details,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions cleanly."""
    logger.warning(f"HTTP Exception [{exc.status_code}] on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code="HTTP_ERROR",
            user_message=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_unhandled_exception_handler(request: Request, exc: Exception):
    """Intercept unhandled exceptions, log technical traceback, return safe 500 error."""
    logger.error(f"Unhandled Exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            user_message="An internal error occurred while processing your request. The technical incident has been logged.",
        ).model_dump(),
    )
