# ── Stage 1: build / dependency resolution ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only the packages needed to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

# Copy sources before building the non-editable package into an isolated prefix.
COPY config/           ./config/
COPY src/              ./src/
COPY scripts/          ./scripts/
COPY evals/            ./evals/
COPY app.py            ./app.py
COPY .env.example      ./.env.example

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-warn-script-location ".[dev]" --no-cache-dir

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for the application
RUN useradd --create-home --uid 1001 appuser

WORKDIR /app

# Python hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Copy pre-built packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY pyproject.toml README.md ./
COPY config/           ./config/
COPY src/              ./src/
COPY scripts/          ./scripts/
COPY evals/            ./evals/
COPY app.py            ./app.py
COPY .env.example      ./.env.example

# Create persistent data directories owned by appuser
RUN mkdir -p data/traces data/evals logs \
    && chown -R appuser:appuser /app

USER appuser

# Seed SQLite databases at image build time so the container starts instantly.
# The seed script is idempotent – re-running it on a volume-mounted data/ directory
# that already has data is safe (it skips existing rows).
RUN python -m scripts.seed_data

# FastAPI (REST + /docs)  → 8000
# Streamlit Inspector UI  → 8501
EXPOSE 8000 8501

# Default: serve the FastAPI backend.
# Override with:  docker run ... streamlit run app.py --server.port=8501 --server.address=0.0.0.0
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
