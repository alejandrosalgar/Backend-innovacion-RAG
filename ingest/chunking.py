"""Chunking por página con solapamiento (objetivo ~400 tokens, ~1500 caracteres)."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    pagina_inicio: int
    pagina_fin: int


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page_text(
    page_text: str,
    page_num: int,
    *,
    target_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[TextChunk]:
    """Divide el texto de una página en fragmentos con solapamiento."""
    text = _normalize_whitespace(page_text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[TextChunk] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= target_chars:
            buf = f"{buf}\n\n{para}".strip() if buf else para
            continue
        if buf:
            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=buf,
                    pagina_inicio=page_num,
                    pagina_fin=page_num,
                )
            )
        if len(para) <= target_chars:
            buf = para
        else:
            for i in range(0, len(para), target_chars - overlap_chars):
                piece = para[i : i + target_chars]
                if piece.strip():
                    chunks.append(
                        TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=piece.strip(),
                            pagina_inicio=page_num,
                            pagina_fin=page_num,
                        )
                    )
            buf = ""
    if buf:
        chunks.append(
            TextChunk(
                chunk_id=str(uuid.uuid4()),
                text=buf,
                pagina_inicio=page_num,
                pagina_fin=page_num,
            )
        )

    # Fusionar fragmentos muy cortos con el siguiente (misma página)
    merged: list[TextChunk] = []
    min_chars = 200
    for ch in chunks:
        if merged and len(ch.text) < min_chars:
            prev = merged[-1]
            merged[-1] = TextChunk(
                chunk_id=prev.chunk_id,
                text=f"{prev.text}\n\n{ch.text}".strip(),
                pagina_inicio=prev.pagina_inicio,
                pagina_fin=ch.pagina_fin,
            )
        else:
            merged.append(ch)
    return merged
