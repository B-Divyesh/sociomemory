"""
SocioMemory - Memory microservice for Sociobot

A standalone memory service with:
- FSRS-optimized retrieval (Free Spaced Repetition Scheduler)
- Temporal knowledge graphs (facts with valid_from/valid_until)
- Memory relationships (updates, extends, derives, contradicts)
- pgvector for similarity search
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sociomemory.api.v1 import router as v1_router
from sociomemory.config import get_settings
from sociomemory.db.session import get_engine, close_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    settings = get_settings()
    logger.info(f"Starting SocioMemory v{settings.version}")
    logger.info(f"Database: {settings.database_url[:50]}...")
    logger.info(f"Environment: {settings.environment}")

    yield

    # Cleanup
    logger.info("Shutting down SocioMemory")
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="SocioMemory",
        description=(
            "Memory microservice for Sociobot with FSRS-optimized retrieval, "
            "temporal knowledge graphs, and memory relationships."
        ),
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(v1_router, prefix="/api")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "sociomemory",
            "version": settings.version,
        }

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with service info."""
        return {
            "service": "SocioMemory",
            "version": settings.version,
            "description": "Memory microservice with FSRS-optimized retrieval",
            "docs": "/docs" if settings.environment != "production" else "disabled",
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "type": type(exc).__name__,
            }
        )

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "sociomemory.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.environment == "development",
        log_level="info",
    )
