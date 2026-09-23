# syntax=docker/dockerfile:1.7
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
COPY chromadb-1.5.9-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu \
    && pip install ./chromadb-1.5.9-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]