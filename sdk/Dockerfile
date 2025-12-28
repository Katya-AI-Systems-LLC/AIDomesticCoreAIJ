# AIPlatform SDK Docker Image
# Multi-stage build for optimized production image

# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml README.md ./
COPY sdk/ ./sdk/

# Install SDK
RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install SDK
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Create non-root user
RUN useradd -m -u 1000 aiplatform
USER aiplatform

# Default command
CMD ["python", "-c", "import sdk; print(f'AIPlatform SDK v{sdk.__version__}')"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sdk" || exit 1

# Labels
LABEL org.opencontainers.image.title="AIPlatform SDK"
LABEL org.opencontainers.image.description="Quantum Infrastructure Zero SDK"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="REChain Network Solutions"
LABEL org.opencontainers.image.licenses="Apache-2.0"
