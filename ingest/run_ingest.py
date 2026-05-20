"""
Vectoriza PDFs del corpus y guarda el índice en disco (ChromaDB).

Uso (desde la carpeta backend, con venv activo):
  python -m ingest.run_ingest

Variables de entorno:
  PMDI_CORPUS_PATH  Ruta a la carpeta con los PDF (por defecto: ../documentos)
  PMDI_CHROMA_PATH  Donde persistir Chroma (por defecto: ./data/chroma)
  PMDI_EMBED_MODEL  Modelo sentence-transformers (multilingüe por defecto)
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from app.settings import get_settings

# Raíz del paquete backend (directorio que contiene app/ e ingest/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _slug_doc_id(stem: str) -> str:
    s = stem.lower().strip()
    s = re.sub(r"[^a-z0-9áéíóúñü]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:120] if s else "documento"


def doc_id_from_filename(filename: str) -> str:
    """
    IDs alineados con el front (anexo-1 … anexo-9, pmdi-v4).
    Si no coincide, se usa un slug del nombre del archivo.
    """
    base = Path(filename).stem
    m = re.match(r"(?i)anexo\s*(\d+)", base)
    if m:
        return f"anexo-{int(m.group(1))}"
    if re.search(r"(?i)PMDI", base) and re.search(r"(?i)Completo|V4|v4", base):
        return "pmdi-v4"
    if re.search(r"(?i)^PMDI\b", base):
        return "pmdi-v4"
    return _slug_doc_id(base)


def main() -> int:
    settings = get_settings()
    if settings.vector_provider != "chroma":
        print(
            "La ingestión implementada actualmente es para Chroma local. "
            "Define PMDI_VECTOR_PROVIDER=chroma para ejecutar este script.",
            file=sys.stderr,
        )
        return 1

    corpus = Path(os.environ.get("PMDI_CORPUS_PATH", BACKEND_ROOT.parent / "documentos")).resolve()
    chroma_path = Path(os.environ.get("PMDI_CHROMA_PATH", BACKEND_ROOT / "data" / "chroma")).resolve()
    model_name = os.environ.get(
        "PMDI_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    if not corpus.is_dir():
        print(f"No existe la carpeta de corpus: {corpus}", file=sys.stderr)
        return 1

    pdfs = sorted(corpus.glob("*.pdf"))
    if not pdfs:
        print(f"No hay PDFs en {corpus}", file=sys.stderr)
        return 1

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as e:
        print("Instala dependencias: pip install -r requirements.txt", file=sys.stderr)
        raise e

    from ingest.chunking import chunk_page_text
    from ingest.extract_pdf import extract_pages

    print(f"Corpus: {corpus}")
    print(f"Chroma: {chroma_path}")
    print(f"Modelo embeddings: {model_name}")
    print(f"PDFs encontrados: {len(pdfs)}")

    chroma_path.mkdir(parents=True, exist_ok=True)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(path=str(chroma_path))

    collection = client.get_or_create_collection(
        name="pmdi_corpus",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )

    # Reconstruir colección desde cero (ingestión completa)
    try:
        client.delete_collection("pmdi_corpus")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="pmdi_corpus",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )

    total_chunks = 0
    for pdf_path in pdfs:
        doc_nombre = pdf_path.name
        doc_id = doc_id_from_filename(pdf_path.name)
        print(f"  -> {doc_nombre} [{doc_id}]")

        try:
            pages = extract_pages(pdf_path)
        except Exception as ex:
            print(f"     ERROR lectura: {ex}", file=sys.stderr)
            continue

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for page_num, page_text in pages:
            for ch in chunk_page_text(page_text, page_num):
                h = hashlib.sha256(ch.text.encode("utf-8")).hexdigest()[:16]
                ids.append(ch.chunk_id)
                documents.append(ch.text)
                metadatas.append(
                    {
                        "doc_id": doc_id,
                        "doc_nombre": doc_nombre[:500],
                        "pagina_inicio": int(ch.pagina_inicio),
                        "pagina_fin": int(ch.pagina_fin),
                        "hash_contenido": h,
                        "capitulo": "",
                        "seccion": "",
                    }
                )

        if not ids:
            print("     (sin texto extraíble)")
            continue

        batch_size = 64
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        total_chunks += len(ids)
        print(f"     fragmentos: {len(ids)}")

    count = collection.count()
    print(f"Listo. Fragmentos en índice: {count} (total_chunks procesados: {total_chunks})")
    print("\nIDs de documento (filtro API / front):")
    seen: set[str] = set()
    for pdf_path in pdfs:
        did = doc_id_from_filename(pdf_path.name)
        if did not in seen:
            seen.add(did)
            print(f"  - {did}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
