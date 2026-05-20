"""
API backend del agente RAG — Plan Maestro PMDI.

Diseño local-first, cloud-ready:
- Proveedor vectorial configurable (hoy: Chroma local)
- Generador de respuesta configurable (hoy: plantilla local)
- Futuro Azure por variables de entorno, sin refactor grande.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from .answering import get_answer_generator
from .schemas import ChatRequest, ChatResponse, HealthResponse
from .settings import get_settings
from .vector_store import get_retriever

_settings = get_settings()
CORS_ORIGINS = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "https://innovacion-demo.web.app",
    "https://innovacion-demo.firebaseapp.com",
]
if _settings.cors_origins_extra.strip():
    for origin in _settings.cors_origins_extra.split(","):
        o = origin.strip()
        if o and o not in CORS_ORIGINS:
            CORS_ORIGINS.append(o)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _settings.warmup_on_start:
        t0 = time.perf_counter()
        retriever, reason = get_retriever()
        if retriever is not None:
            get_answer_generator()
            logger.info(
                "Warmup OK: %s, %d chunks (%.1f s)",
                retriever.provider_name,
                retriever.count,
                time.perf_counter() - t0,
            )
        else:
            logger.warning("Warmup falló: %s", reason)
    yield


app = FastAPI(
    title="PMDI — Agente RAG",
    description="Consulta documental institucional (Plan Maestro Medellín Inteligente).",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _llm_status() -> tuple[str, bool]:
    settings = get_settings()
    configured = settings.llm_provider
    try:
        generator, _ = get_answer_generator()
        active = generator.provider_name == "azure-openai"
    except Exception:
        active = False
    return configured, active


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    llm_configured, llm_active = _llm_status()
    retriever, reason = get_retriever()
    if retriever is not None:
        return HealthResponse(
            corpus_version=f"{retriever.provider_name}-indexed",
            indexed_chunks=retriever.count,
            vector_store=retriever.provider_name,
            llm_provider_configured=llm_configured,
            llm_active=llm_active,
        )
    return HealthResponse(
        corpus_version="mock-v1",
        indexed_chunks=None,
        vector_store=reason or "no disponible",
        llm_provider_configured=llm_configured,
        llm_active=llm_active,
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    question = body.messages[-1].content.strip()

    if "no hay" in question.lower() or "sin informacion" in question.lower():
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ChatResponse(
            answer=(
                "No dispongo de informacion en el corpus documental para responder "
                "esa consulta con la informacion indexada actualmente."
            ),
            sources=[],
            confidence="low",
            insufficient_context=True,
            latency_ms=elapsed,
        )

    retriever, retriever_reason = get_retriever()
    if retriever is None:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ChatResponse(
            answer=(
                "No hay índice vectorial disponible. "
                f"Detalle: {retriever_reason or 'sin detalle'}\n\n"
                "Modo local esperado: PMDI_VECTOR_PROVIDER=chroma y ejecutar "
                "`python -m ingest.run_ingest`."
            ),
            sources=[],
            confidence="low",
            insufficient_context=True,
            latency_ms=elapsed,
        )

    sources = retriever.query(question, body.doc_id)
    if not sources:
        elapsed = int((time.perf_counter() - t0) * 1000)
        hint = (
            "Prueba una pregunta más concreta (p. ej. objetivos del PMDI, OKRs, gobernanza o un anexo)."
            if body.doc_id is None
            else f"No hay fragmentos para doc_id={body.doc_id!r}. Verifica el ID indexado en la ingestión."
        )
        return ChatResponse(
            answer=(
                "No encontré fragmentos del corpus con suficiente relevancia para esta pregunta. "
                f"{hint}"
            ),
            sources=[],
            confidence="low",
            insufficient_context=True,
            latency_ms=elapsed,
        )

    generator, _generator_reason = get_answer_generator()
    answer = generator.generate(question, sources)

    confidence = "high" if sources[0].score >= 0.8 else "medium"
    elapsed = int((time.perf_counter() - t0) * 1000)
    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,  # type: ignore[arg-type]
        insufficient_context=False,
        latency_ms=elapsed,
        answer_provider=generator.provider_name,
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
