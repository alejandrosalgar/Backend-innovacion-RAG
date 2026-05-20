from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from .schemas import SourceChunk
from .settings import get_settings


class AnswerGenerator(Protocol):
    provider_name: str

    def generate(self, question: str, sources: list[SourceChunk]) -> str: ...


class LocalAnswerGenerator:
    """Respuestas cortas y directas; el detalle documental va en el panel de fuentes de la UI."""

    provider_name = "local-template"

    def generate(self, question: str, sources: list[SourceChunk]) -> str:
        if not sources:
            return (
                "No encontré en el plan algo que responda bien a eso. "
                "Prueba con otra redacción o algo más concreto (por ejemplo, un anexo u objetivo del PMDI)."
            )

        qn = _normalize(question)
        answer = self._direct_answer(qn, question, sources)

        if _wants_literal_quote(question):
            quotes = _literal_quotes_block(sources[:2])
            if quotes:
                return f"{answer}\n\n{quotes}"

        return answer

    def _direct_answer(self, qn: str, question: str, sources: list[SourceChunk]) -> str:
        if _is_yes_no_style(qn) and _is_scope_question(qn):
            if _sources_support_medellin_scope(sources):
                return (
                    "Sí, aplica a Medellín. El Plan Maestro Medellín Inteligente está formulado "
                    "para el distrito y su contexto en Colombia; no es un documento genérico sin "
                    "ese enfoque territorial."
                )
            return (
                "Con los pasajes que recuperé no alcanza para afirmarlo con claridad. "
                "Revisa las fuentes de abajo o precisa a qué sección te refieres con «esto»."
            )

        if _is_okr_question(qn):
            return (
                "Sí: el plan trabaja OKRs e indicadores (metas, seguimiento y tablas de cumplimiento "
                "en niveles estratégico y táctico). El detalle está en las fuentes citadas abajo."
            )

        if _is_document_overview(qn):
            doc = _short_doc_name(sources[0])
            if len({s.doc_id for s in sources[:3]}) == 1:
                return (
                    f"En «{doc}» aparecen temas de contexto, lineamientos y desarrollo del PMDI "
                    f"(págs. {sources[0].pagina_inicio}–{sources[0].pagina_fin} y siguientes). "
                    "Abajo tienes los extractos; si quieres el texto palabra por palabra, pídeme la cita literal."
                )
            return (
                "En el corpus del plan hay varios apartados relacionados con tu pregunta "
                "(contexto, lineamientos, referentes e implementación). "
                "Mira las fuentes abajo; para texto exacto del PDF, pide «cita literal»."
            )

        if _is_yes_no_style(qn):
            return (
                "Según lo indexado del PMDI, hay material relacionado, pero tu pregunta es muy abierta "
                "para un sí o no rotundo. Revisa las fuentes abajo o concreta qué apartado te interesa."
            )

        return (
            "Con lo que hay en el plan sobre este tema, lo más útil es revisar las fuentes que aparecen "
            "abajo. Respondo en pocas líneas; si necesitas el párrafo exacto del documento, "
            "escribe «dame la cita literal»."
        )


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def _clean_excerpt(text: str, *, max_len: int = 480) -> str:
    t = unicodedata.normalize("NFKC", text.replace("\n", " "))
    t = re.sub(r"\|{2,}", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    return t


def _short_doc_name(src: SourceChunk) -> str:
    name = (src.doc_nombre or "documento del PMDI").strip()
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    if len(name) > 72:
        return name[:69].rstrip() + "…"
    return name


def _cite_ref(src: SourceChunk) -> str:
    return f"{_short_doc_name(src)}, pág. {src.pagina_inicio}–{src.pagina_fin}"


def _wants_literal_quote(question: str) -> bool:
    qn = _normalize(question)
    return any(
        k in qn
        for k in (
            "cita literal",
            "texto literal",
            "texto exacto",
            "extracto",
            "copia el texto",
            "copiar el texto",
            "que dice exactamente",
            "palabra por palabra",
            "transcribe",
            "transcripcion",
        )
    )


def _literal_quotes_block(sources: list[SourceChunk]) -> str:
    if not sources:
        return ""
    parts: list[str] = []
    for src in sources:
        body = _clean_excerpt(src.excerpt)
        if not body:
            continue
        parts.append(f"«{body}»\n— {_cite_ref(src)}")
    if not parts:
        return ""
    return "Texto del documento:\n\n" + "\n\n".join(parts)


def _is_yes_no_style(qn: str) -> bool:
    return any(
        k in qn
        for k in (
            "aplica",
            "aplicable",
            "es para",
            "solo para",
            "unicamente",
            "cierto que",
            "es verdad",
            "puedo usar",
        )
    ) or qn.startswith(("es ", "esta ", "esto "))


def _is_document_overview(qn: str) -> bool:
    return any(
        k in qn
        for k in (
            "que tiene",
            "que contiene",
            "que incluye",
            "de que trata",
            "de que habla",
            "este documento",
            "el documento",
            "contenido del",
            "resume",
            "resumen",
            "habla de",
            "trata de",
            "que es el",
        )
    )


def _is_scope_question(qn: str) -> bool:
    return any(k in qn for k in ("medellin", "colombia", "alcance", "territorio", "distrito"))


def _is_okr_question(qn: str) -> bool:
    return any(k in qn for k in ("okr", "okrs", "indicador", "indicadores"))


def _sources_support_medellin_scope(sources: list[SourceChunk]) -> bool:
    for src in sources[:5]:
        txt = _normalize(src.excerpt)
        if "medellin" in txt and any(k in txt for k in ("plan maestro", "distrito", "ciudad", "colombia")):
            return True
    return False


_AZURE_SYSTEM_PROMPT = """Eres un asistente del Plan Maestro Medellín Inteligente (PMDI).

Estilo:
- Responde como en un chat: primero la respuesta directa en 1 a 4 oraciones cortas.
- No pegues párrafos largos del contexto ni reescribas trozos extensos del PDF.
- No parafrasees bloques del documento salvo que el usuario pida explícitamente cita literal, extracto o texto exacto.
- Si piden cita literal, incluye solo el extracto necesario entre comillas y la página.
- Usa solo el CONTEXTO; si falta información, dilo sin inventar.
- Puedes mencionar el documento y la página entre paréntesis una vez si ayuda, pero no hagas listas de referencias al final (la interfaz ya muestra las fuentes).
- Español claro, sin jerga (RAG, embeddings, fragmentos)."""


class AzureOpenAIAnswerGenerator:
    provider_name = "azure-openai"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            raise RuntimeError("Faltan AZURE_OPENAI_ENDPOINT o AZURE_OPENAI_API_KEY.")
        if not settings.azure_openai_chat_deployment:
            raise RuntimeError("Falta AZURE_OPENAI_CHAT_DEPLOYMENT.")
        try:
            from openai import AzureOpenAI
        except Exception as ex:
            raise RuntimeError("Paquete openai no instalado. Ejecuta pip install -r requirements.txt") from ex

        self._deployment = settings.azure_openai_chat_deployment
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    def generate(self, question: str, sources: list[SourceChunk]) -> str:
        context_lines: list[str] = []
        for src in sources[:5]:
            context_lines.append(
                f"[{_short_doc_name(src)} | pág. {src.pagina_inicio}–{src.pagina_fin}]\n"
                f"{_clean_excerpt(src.excerpt, max_len=1200)}"
            )
        context = "\n\n".join(context_lines)
        literal = _wants_literal_quote(question)
        user_prompt = (
            f"Pregunta:\n{question}\n\n"
            f"Modo cita literal solicitado: {'sí' if literal else 'no'}\n\n"
            f"Contexto:\n{context}"
        )
        resp = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0.25,
            max_tokens=450,
            messages=[
                {"role": "system", "content": _AZURE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


_answer_generator: AnswerGenerator | None = None
_answer_generator_fallback: str | None = None


def get_answer_generator() -> tuple[AnswerGenerator, str | None]:
    global _answer_generator, _answer_generator_fallback
    if _answer_generator is not None:
        return _answer_generator, _answer_generator_fallback

    settings = get_settings()
    if settings.llm_provider == "azure_openai":
        try:
            _answer_generator = AzureOpenAIAnswerGenerator()
            _answer_generator_fallback = None
        except Exception as ex:
            _answer_generator = LocalAnswerGenerator()
            _answer_generator_fallback = f"azure_openai no disponible: {ex}"
    else:
        _answer_generator = LocalAnswerGenerator()
        _answer_generator_fallback = None
    return _answer_generator, _answer_generator_fallback
