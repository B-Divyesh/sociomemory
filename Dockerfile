# SocioMemory Production Dockerfile
# Multi-stage build for optimized image size

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only pyproject.toml first for dependency caching
COPY pyproject.toml .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the package in non-editable mode
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# v17: Pre-download flashrank cross-encoder model at build time
# This avoids runtime download latency and ensures deterministic container behavior
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/opt/flashrank_cache')"

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# v17: Copy pre-downloaded flashrank model from builder
COPY --from=builder /opt/flashrank_cache /opt/flashrank_cache

# Copy application code
COPY sociomemory/ /app/sociomemory/
COPY scripts/ /app/scripts/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app /opt/flashrank_cache

USER appuser

# Azure App Service uses PORT environment variable
# Azure App Service expects port 80 for Linux containers by default
ENV PORT=80

# Expose port 80 for Azure
EXPOSE 80

# Health check for Azure App Service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
# Use shell form to allow PORT variable expansion
CMD uvicorn sociomemory.main:app --host 0.0.0.0 --port ${PORT}
