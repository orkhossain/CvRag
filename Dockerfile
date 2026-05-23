FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN useradd -m -u 1000 user

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY --chown=user . /app
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "uvicorn", "cvrag.main:app", "--host", "0.0.0.0", "--port", "7860"]
