FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_HTTP_TIMEOUT=300

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY fixtures ./fixtures
COPY scripts ./scripts

RUN uv venv
RUN uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
RUN uv export --frozen --no-dev --format requirements-txt --no-hashes \
    | grep -Ev '^(torch|nvidia-|cuda-|triton)' > /tmp/requirements.txt \
    && uv pip install --python .venv/bin/python -r /tmp/requirements.txt

EXPOSE 8000
CMD [".venv/bin/uvicorn", "rag_permission.api:app", "--host", "0.0.0.0", "--port", "8000"]
