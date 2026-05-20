"""Esquemas alineados al plan técnico (citas, confianza, trazabilidad)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    """Mensaje entrante del usuario."""

    role: Literal["user"] = "user"
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """Cuerpo POST /api/v1/chat — Etapa A del flujo (recepción)."""

    messages: list[ChatMessageIn] = Field(..., min_length=1)
    conversation_id: str | None = None
    doc_id: str | None = Field(
        default=None,
        description="Filtro opcional por documento (namespace / doc_id).",
    )


class SourceChunk(BaseModel):
    """Fragmento recuperado con metadatos para citación (plan secc. 4.3 y 5.E)."""

    chunk_id: str
    doc_id: str
    doc_nombre: str
    capitulo: str | None = None
    seccion: str | None = None
    pagina_inicio: int
    pagina_fin: int
    excerpt: str = Field(..., description="Texto citado del fragmento.")
    score: float = Field(..., ge=0.0, le=1.0, description="Similitud / confianza de recuperación.")


class ChatResponse(BaseModel):
    """Respuesta del agente con fuentes obligatorias cuando hay respuesta factual."""

    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    insufficient_context: bool = False
    latency_ms: int | None = None
    answer_provider: str = Field(
        default="local-template",
        description="Motor de redacción: local-template (sin LLM) o azure-openai.",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "pmdi-rag-api"
    corpus_version: str = "mock-v1"
    indexed_chunks: int | None = None
    vector_store: str | None = Field(
        default=None,
        description="Estado del almacén vectorial (chromadb, vacío, error).",
    )
    llm_provider_configured: str = Field(
        default="local",
        description="Valor de PMDI_LLM_PROVIDER (local | azure_openai).",
    )
    llm_active: bool = Field(
        default=False,
        description="True si el generador activo es Azure OpenAI y está operativo.",
    )
