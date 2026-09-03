FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /code

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
COPY importer ./importer
RUN uv sync --frozen --no-dev

FROM python:3.11-slim

WORKDIR /code

COPY --from=builder /code/.venv /code/.venv
COPY app ./app
COPY importer ./importer

ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
