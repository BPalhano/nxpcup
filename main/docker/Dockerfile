# ==============================================================================
# NXP Cup Autonomous Driving - Production Multi-Stage Dockerfile
# Optimized for Embedded Edge Systems (x86_64 and ARM64 / Jetson / RPi5)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=0

WORKDIR /build

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency resolution
RUN pip install --no-cache-dir uv

# Create isolated virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pyproject configuration
COPY pyproject.toml /build/

# Install python dependencies inside virtual environment
RUN uv pip install --no-cache \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    pydantic>=2.5.0 \
    python-multipart>=0.0.9 \
    onnx>=1.15.0 \
    onnxruntime>=1.17.0 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.26.0,<2.0.0 \
    pandas>=2.0.0 \
    pillow>=10.0.0 \
    pyyaml>=6.0.1 \
    timm>=0.9.12 \
    albumentations>=1.4.0

# ------------------------------------------------------------------------------
# Stage 2: Runtime (Minimal Production Image)
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:/app/main"

WORKDIR /app

# Install minimal dynamic libraries required for OpenCV and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-root user
RUN groupadd -g 1000 nxpuser && \
    useradd -u 1000 -g nxpuser -s /bin/bash -m nxpuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source code and models
COPY main/src /app/main/src
COPY main/configs /app/main/configs
COPY main/training_results /app/main/training_results
COPY pyproject.toml /app/

# Set file permissions for non-root execution
RUN chown -R nxpuser:nxpuser /app

USER nxpuser

# Expose FastAPI service port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "main.src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
