FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://claude.ai/install.sh | bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/
RUN uv sync --no-dev --frozen

COPY migrations/ ./migrations/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV MODE=slack
ENV PATH="/root/.local/bin:$PATH"

ENTRYPOINT ["./entrypoint.sh"]
