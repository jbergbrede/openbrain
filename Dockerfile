FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
RUN uv sync --no-dev --frozen

COPY migrations/ ./migrations/

ENV MODE=slack

CMD ["sh", "-c", "uv run openbrain --mode $MODE"]
