# Stage 1: build virtual environment with uv from uv.lock
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN uv sync --frozen --no-dev

# Stage 2: minimal non-root runtime image
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH="/app/data/index.bin" \
    PORT="8000"

RUN groupadd --system --gid 1001 findex && \
    useradd --system --uid 1001 --gid findex --create-home findex

WORKDIR /app

COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/src /app/src
COPY --chown=findex:findex data/index.bin /app/data/index.bin

USER findex

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn findex.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
