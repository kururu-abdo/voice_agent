# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

FROM base AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /app/.venv && \
    /app/.venv/bin/python -m pip install --upgrade pip && \
    /app/.venv/bin/python -m pip install .

FROM base
ARG UID=10001
RUN adduser --disabled-password --gecos "" --home /app --shell /sbin/nologin --uid ${UID} appuser
WORKDIR /app
COPY --from=build --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser documents ./documents
COPY --chown=appuser:appuser pyproject.toml README.md ./
RUN mkdir -p /app/data/chroma /app/uploads && chown -R appuser:appuser /app/data /app/uploads
ENV PATH="/app/.venv/bin:$PATH"
USER appuser
RUN python -m src.agent download-files
CMD ["python", "-m", "src.agent", "start"]
