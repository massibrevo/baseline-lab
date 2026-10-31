FROM node:22.12.0-bookworm-slim AS web
WORKDIR /web
COPY apps/web/package*.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

FROM python:3.12.14-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_LINK_MODE=copy BASELINE_DATA_DIR=/data
RUN groupadd -g 10001 baseline && useradd -m -u 10001 -g baseline baseline && mkdir /data && chown baseline:baseline /data
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev && chown -R baseline:baseline /app
COPY --from=web /web/dist /app/apps/web/dist
USER baseline
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "baseline_lab.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
