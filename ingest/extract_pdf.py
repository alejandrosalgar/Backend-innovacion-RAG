"""Extracción de texto por página (PyMuPDF)."""

from __future__ import annotations

from pathlib import Path


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Devuelve lista (número_página_1_based, texto)."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    try:
        pages: list[tuple[int, str]] = []
        for i in range(len(doc)):
            text = doc[i].get_text("text") or ""
            pages.append((i + 1, text))
        return pages
    finally:
        doc.close()
