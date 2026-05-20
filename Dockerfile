# API FastAPI + índice Chroma embebido (copiar data/chroma antes del build desde la máquina donde corrida la ingestión).
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Carpeta generada por `python -m ingest.run_ingest` (no suele estar en git).
COPY data/chroma ./data/chroma

ENV PMDI_API_HOST=0.0.0.0
ENV PMDI_VECTOR_PROVIDER=chroma
ENV PMDI_CHROMA_PATH=/app/data/chroma
ENV PMDI_API_RELOAD=0

EXPOSE 8080
ENV PORT=8080

# Cloud Run inyecta PORT; Hosting proxy llama a /api/* en el mismo origen.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
