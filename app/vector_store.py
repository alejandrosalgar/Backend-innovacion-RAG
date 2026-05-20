from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from .schemas import SourceChunk
from .settings import get_settings

COLLECTION = "pmdi_corpus"
TOP_K = 5

_STOPWORDS = {
    "de",
    "la",
    "el",
    "los",
    "las",
    "y",
    "o",
    "en",
    "del",
    "al",
    "un",
    "una",
    "que",
    "es",
    "se",
    "para",
    "por",
    "con",
    "esto",
    "solo",
}


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_set(text: str) -> set[str]:
    toks = [t for t in _normalize_text(text).split() if len(t) > 2 and t not in _STOPWORDS]
    return set(toks)


def _lexical_overlap(question_tokens: set[str], excerpt: str) -> float:
    if not question_tokens:
        return 0.0
    ex_tokens = _token_set(excerpt)
    if not ex_tokens:
        return 0.0
    return len(question_tokens.intersection(ex_tokens)) / max(1, len(question_tokens))


def _embedding_query_text(question: str) -> str:
    """Enriquece preguntas vagas para mejorar el embedding semántico."""
    qn = _normalize_text(question)
    if len(qn.split()) <= 8 and any(
        k in qn for k in ("documento", "documentos", "contiene", "tiene", "habla", "trata")
    ):
        return (
            f"{question.strip()} — contenido, temas y secciones del "
            "Plan Maestro Medellín Inteligente (PMDI)."
        )
    return question


def _query_variants(question: str, *, extra_variants: bool) -> list[str]:
    if not extra_variants:
        return [question]

    qn = _normalize_text(question)
    variants = [question]

    if any(k in qn for k in ("indicador", "indicadores", "okr", "okrs")):
        variants.append("indicadores y okrs del plan maestro medellin inteligente")
    if any(k in qn for k in ("solo", "unicamente", "aplica", "medellin", "colombia", "alcance")):
        variants.append("alcance geografico del plan maestro medellin inteligente colombia")

    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


_chroma_collection = None


def _get_chroma_collection():
    """Cliente + colección + modelo de embeddings (una sola carga por proceso)."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb
    from chromadb.utils import embedding_functions

    settings = get_settings()
    if not settings.chroma_abs_path.is_dir():
        raise RuntimeError(f"Chroma no inicializado en {settings.chroma_abs_path}")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embed_model)
    client = chromadb.PersistentClient(path=str(settings.chroma_abs_path))
    collection = client.get_collection(name=COLLECTION, embedding_function=ef)
    _chroma_collection = collection
    return collection


class BaseRetriever(Protocol):
    provider_name: str

    @property
    def count(self) -> int: ...

    def query(self, question: str, doc_id: str | None = None, *, top_k: int = TOP_K) -> list[SourceChunk]: ...


class ChromaRetriever:
    provider_name = "chroma"

    def __init__(self) -> None:
        self._collection = _get_chroma_collection()

    @property
    def count(self) -> int:
        return int(self._collection.count())

    def _build_chunks(
        self,
        ranked: list[tuple[str, tuple[float, float, dict, str]]],
        *,
        top_k: int,
        min_combined: float,
    ) -> list[SourceChunk]:
        out: list[SourceChunk] = []
        for cid, (combined, _vec, meta, text) in ranked:
            if combined < min_combined:
                continue
            out.append(
                SourceChunk(
                    chunk_id=cid,
                    doc_id=str(meta.get("doc_id", "")),
                    doc_nombre=str(meta.get("doc_nombre", "")),
                    capitulo=(meta.get("capitulo") or None) or None,
                    seccion=(meta.get("seccion") or None) or None,
                    pagina_inicio=int(meta.get("pagina_inicio", 1)),
                    pagina_fin=int(meta.get("pagina_fin", 1)),
                    excerpt=text[:2000] + ("..." if len(text) > 2000 else ""),
                    score=round(combined, 4),
                )
            )
            if len(out) >= top_k:
                break
        return out

    def query(self, question: str, doc_id: str | None = None, *, top_k: int = TOP_K) -> list[SourceChunk]:
        settings = get_settings()
        where = {"doc_id": {"$eq": doc_id}} if doc_id else None
        embed_seed = _embedding_query_text(question)
        variants = _query_variants(embed_seed, extra_variants=settings.query_variants_enabled)
        n_results = max(top_k, min(settings.retrieval_n_results, 15))
        res = self._collection.query(
            query_texts=variants,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        if not res["ids"]:
            return []

        q_tokens = _token_set(question)
        # cid -> (combined_score, vector_score, meta, excerpt)
        merged: dict[str, tuple[float, float, dict, str]] = {}
        for q_idx, ids_for_q in enumerate(res["ids"]):
            if not ids_for_q:
                continue
            for i, cid in enumerate(ids_for_q):
                meta = (res["metadatas"][q_idx][i] or {}) if res["metadatas"] else {}
                dist = res["distances"][q_idx][i] if res["distances"] else 0.0
                text = ((res["documents"][q_idx][i] or "") if res["documents"] else "").strip()
                vector_score = max(0.0, min(1.0, 1.0 - float(dist)))
                lex = _lexical_overlap(q_tokens, text)
                combined = (0.78 * vector_score) + (0.22 * lex)
                # Boost para preguntas de alcance geográfico
                qn = _normalize_text(question)
                if any(k in qn for k in ("medellin", "colombia", "alcance")) and (
                    "medellin" in _normalize_text(text) or "colombia" in _normalize_text(text)
                ):
                    combined += 0.06
                combined = min(1.0, combined)

                if vector_score < settings.min_vector_score:
                    continue
                prev = merged.get(cid)
                if prev is None or combined > prev[0]:
                    merged[cid] = (combined, vector_score, meta, text)

        ranked = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
        out = self._build_chunks(ranked, top_k=top_k, min_combined=settings.min_retrieval_score)

        if not out and ranked and settings.retrieval_fallback:
            relaxed = max(0.45, settings.min_retrieval_score - 0.12)
            out = self._build_chunks(ranked, top_k=top_k, min_combined=relaxed)

        return out


class AzureAISearchRetriever:
    provider_name = "azure-ai-search"

    def __init__(self) -> None:
        settings = get_settings()
        missing = []
        if not settings.azure_search_endpoint:
            missing.append("AZURE_SEARCH_ENDPOINT")
        if not settings.azure_search_api_key:
            missing.append("AZURE_SEARCH_API_KEY")
        if not settings.azure_search_index:
            missing.append("AZURE_SEARCH_INDEX")
        if not settings.azure_openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not settings.azure_openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not settings.azure_openai_embedding_deployment:
            missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if missing:
            raise RuntimeError(f"Configuración Azure incompleta: {', '.join(missing)}")
        raise RuntimeError(
            "Proveedor azure_ai_search preparado en configuración, "
            "pero su implementación de query se deja para el siguiente paso."
        )

    @property
    def count(self) -> int:
        return 0

    def query(self, question: str, doc_id: str | None = None, *, top_k: int = TOP_K) -> list[SourceChunk]:
        return []


_retriever: BaseRetriever | None = None


def get_retriever() -> tuple[BaseRetriever | None, str | None]:
    global _retriever
    if _retriever is not None and _retriever.count > 0:
        return _retriever, None

    settings = get_settings()
    try:
        if settings.vector_provider == "azure_ai_search":
            _retriever = AzureAISearchRetriever()
        else:
            _retriever = ChromaRetriever()
        if _retriever.count <= 0:
            return None, "Índice vacío; ejecuta: python -m ingest.run_ingest"
        return _retriever, None
    except Exception as ex:
        _retriever = None
        return None, str(ex)
