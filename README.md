# Backend — Agente RAG PMDI

API en **FastAPI** para consultas sobre el corpus de PDFs del Plan Maestro Medellín Inteligente (PMDI). Incluye **ingestión**, **chunking**, **embeddings** y almacenamiento vectorial en **ChromaDB** (local), con diseño preparado para migrar a **Azure** (OpenAI + AI Search) mediante variables de entorno.

---

## Contenido de esta carpeta

| Ruta | Rol |
|------|-----|
| `app/main.py` | API HTTP: `/health`, `/api/v1/chat` |
| `app/settings.py` | Configuración centralizada (`pydantic-settings`, lee `.env`) |
| `app/schemas.py` | Modelos Pydantic (request/response del chat) |
| `app/vector_store.py` | Recuperación semántica sobre Chroma (y stub Azure AI Search) |
| `app/answering.py` | Generación de respuesta: plantilla local o Azure OpenAI |
| `ingest/run_ingest.py` | **Script principal de vectorización** (PDF → chunks → embeddings → Chroma) |
| `ingest/extract_pdf.py` | Extracción de texto por página con PyMuPDF |
| `ingest/chunking.py` | División del texto en fragmentos (chunks) con solapamiento |
| `run.py` | Arranque del servidor sin escribir `uvicorn` a mano |
| `data/chroma/` | **Persistencia del índice vectorial** (generada al ejecutar la ingestión; no versionar en git) |
| `.env` / `.env.example` | Variables de entorno (copiar ejemplo y ajustar) |

---

## Requisitos previos

- **Python 3.10+** (recomendado 3.11 en entornos nuevos).
- Espacio en disco: el modelo de embeddings y dependencias (PyTorch, etc.) ocupan varios GB la primera vez.
- Los PDFs del corpus en una carpeta (por defecto `../documentos` respecto a `backend/`).

---

## Instalación rápida

Desde el directorio `backend`:

```bash
python -m venv .venv
```

En Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

Copiar configuración:

```bash
copy .env.example .env
```

Editar `.env` según la sección [Configuración](#configuración).

---

## Configuración

El archivo `app/settings.py` carga variables desde el entorno y desde un archivo **`.env`** en la raíz de `backend/` (mismo nivel que `run.py`).

### Variables principales

| Variable | Descripción | Valor típico (local) |
|----------|-------------|----------------------|
| `PMDI_API_HOST` | Host del servidor HTTP | `127.0.0.1` |
| `PMDI_API_PORT` | Puerto | `8080` (en Windows a veces se evita `8000` por conflictos) |
| `PMDI_API_RELOAD` | Recarga automática del código (`1`/`0`) | `0` en Windows si hay errores de socket |
| `PMDI_VECTOR_PROVIDER` | Dónde está el índice vectorial | `chroma` |
| `PMDI_LLM_PROVIDER` | Cómo se genera el texto de respuesta | `local` o `azure_openai` |
| `PMDI_CHROMA_PATH` | Carpeta persistente de Chroma | `./data/chroma` |
| `PMDI_EMBED_MODEL` | Modelo Hugging Face para embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `PMDI_CORPUS_PATH` | *(solo ingestión)* Ruta absoluta o relativa a los PDFs | Por defecto `../documentos` |

Para Azure OpenAI y Azure AI Search, ver comentarios en `.env.example`.

**Importante:** La ingestión solo se ejecuta si `PMDI_VECTOR_PROVIDER=chroma`. Si pones `azure_ai_search` sin haber implementado aún la carga en ese índice, el script `run_ingest` saldrá con error a propósito.

---

## Arrancar la API

Desde `backend` (con el venv activado):

```bash
python run.py
```

Equivalente:

```bash
python -m app.main
```

Documentación interactiva: `http://127.0.0.1:8080/docs` (Swagger).

### Endpoints

- **`GET /health`**  
  Indica si hay índice cargado (`vector_store`), versión lógica del corpus y número de fragmentos indexados (`indexed_chunks`).

- **`POST /api/v1/chat`**  
  Cuerpo JSON mínimo:

  ```json
  {
    "messages": [{ "role": "user", "content": "Tu pregunta" }],
    "doc_id": null
  }
  ```

  - `doc_id: null` o ausente → búsqueda en **todo el corpus**.
  - `doc_id: "anexo-5"` → solo fragmentos con ese metadato (debe coincidir con los IDs generados en la ingestión).

---

## Vectorización: visión general

La **vectorización** es el proceso que:

1. Lee cada PDF del corpus.
2. Extrae texto **por página**.
3. Divide el texto en **chunks** (fragmentos) con tamaño y solapamiento controlados.
4. Asigna a cada chunk un **identificador único** y **metadatos** (`doc_id`, nombre del archivo, páginas, hash del contenido).
5. Pasa cada texto por un modelo de **embeddings** (vectores numéricos que representan el significado).
6. Guarda vectores + metadatos + texto en **ChromaDB** persistente bajo `PMDI_CHROMA_PATH`.

En consulta, el backend **no** vuelve a leer los PDFs: solo consulta el índice y devuelve los fragmentos más similares a la pregunta.

---

## Script de ingestión: paso a paso (`python -m ingest.run_ingest`)

El punto de entrada es `ingest/run_ingest.py`, función `main()`.

### Paso 0 — Comprobación de configuración

1. Se llama a `get_settings()` (`app/settings.py`).
2. Si `PMDI_VECTOR_PROVIDER != "chroma"`, el script imprime un mensaje y termina con código de error **1**.  
   Motivo: hoy la ingestión implementada escribe solo en Chroma local.

### Paso 1 — Rutas y modelo

1. **`corpus`**: carpeta de PDFs.  
   - Por defecto: directorio padre de `backend` + `documentos` → típicamente `../documentos` desde `backend/`.  
   - Se puede forzar con `PMDI_CORPUS_PATH` (ruta absoluta recomendada si mueves el proyecto).

2. **`chroma_path`**: dónde guardar el índice.  
   - Por defecto: `backend/data/chroma` (relativo: `./data/chroma` en `.env`).

3. **`model_name`**: modelo de embeddings.  
   - Por defecto: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.  
   - Sobrescribible con `PMDI_EMBED_MODEL`.

### Paso 2 — Validación del corpus

1. Debe existir el directorio `corpus`.
2. Debe haber al menos un archivo `*.pdf` en ese directorio.

Si falla, el script escribe en **stderr** y sale con código **1**.

### Paso 3 — Dependencias de Chroma y Sentence Transformers

1. Importa `chromadb` y `chromadb.utils.embedding_functions`.
2. Si falla, indica instalar `requirements.txt`.

### Paso 4 — Inicialización de Chroma y función de embedding

1. Se crea `SentenceTransformerEmbeddingFunction(model_name=...)`.  
   - La **primera ejecución** descarga el modelo desde Hugging Face (puede tardar y ocupar espacio).
2. `PersistentClient(path=chroma_path)` abre o crea la base en disco.
3. Colección: nombre fijo **`pmdi_corpus`**, metadata `{"hnsw:space": "cosine"}` (similitud coseno en el espacio de embeddings).

### Paso 5 — Reconstrucción completa del índice

1. Se intenta `delete_collection("pmdi_corpus")` para evitar mezclar datos viejos con nuevos.
2. Se vuelve a crear la colección vacía con la misma configuración.

**Efecto:** cada ejecución completa de `run_ingest` **reemplaza** el índice anterior para esa colección. Si necesitas versionado histórico, haz copia de `data/chroma` o cambia `PMDI_CHROMA_PATH` por versión.

### Paso 6 — Por cada archivo PDF

Para cada `pdf_path` en orden alfabético (`sorted(corpus.glob("*.pdf"))`):

#### 6.1 Identificador lógico del documento (`doc_id`)

Función `doc_id_from_filename` (`run_ingest.py`):

- Si el nombre del archivo (sin extensión) coincide desde el inicio con `anexo` + número (insensible a mayúsculas, espacios opcionales) → `doc_id = "anexo-N"` (ej. `anexo-1`, `anexo-9`).
- Si el nombre sugiere PMDI completo (presencia de `PMDI` y `Completo`/`V4`, o empieza por `PMDI`) → `doc_id = "pmdi-v4"`.
- En cualquier otro caso → **slug** del nombre: minúsculas, caracteres no alfanuméricos sustituidos por guiones, máximo 120 caracteres.

Este `doc_id` debe alinearse con el **filtro del frontend/API** (`doc_id` en `POST /api/v1/chat`).

#### 6.2 Extracción de texto (`ingest/extract_pdf.py`)

- Abre el PDF con **PyMuPDF** (`fitz`).
- Para cada página `i` (0-based en el motor):  
  `text = doc[i].get_text("text") or ""`
- Devuelve una lista de tuplas `(número_de_página_1_based, texto)`.

**Limitaciones habituales:** PDFs con muchas anotaciones pueden mostrar avisos de MuPDF en consola; el texto plano puede seguir extrayéndose. PDFs escaneados sin OCR pueden dar páginas vacías.

#### 6.3 Chunking por página (`ingest/chunking.py`)

Para cada par `(page_num, page_text)`:

- Función **`chunk_page_text(page_text, page_num, ...)`**.

Detalle del algoritmo:

1. **Normalización de espacios** (`_normalize_whitespace`): unifica saltos de línea, colapsa espacios, recorta.
2. Si el texto queda vacío → no hay chunks para esa página.
3. **División en párrafos**: split por bloques separados por líneas en blanco (`\n\s*\n`). Si no hay párrafos, se usa la página entera como un solo bloque.
4. **Acumulación en buffer** hasta un tamaño objetivo:
   - `target_chars` por defecto **1500** (aprox. orden de magnitud ~400 tokens según el plan técnico; no es conteo exacto de tokens).
   - Si un párrafo supera el límite, se **corta en ventanas** con paso `target_chars - overlap_chars` (por defecto solapamiento **200** caracteres entre ventanas consecutivas).
5. Cada chunk recibe un **`chunk_id`** único (UUID v4).
6. **`pagina_inicio` y `pagina_fin`**: en esta implementación, por construcción, ambas coinciden con el número de página actual (chunking **por página**; no se cruza texto entre páginas en un mismo chunk).
7. **Post-proceso**: fragmentos muy cortos (< 200 caracteres) se **fusionan con el fragmento anterior** en la misma página (mismo `chunk_id` que el bloque previo) para reducir ruido en recuperación.

#### 6.4 Metadatos y listas para Chroma

Por cada `TextChunk`:

- `ids.append(chunk_id)` — id del vector/fila en Chroma.
- `documents.append(texto_del_chunk)` — lo que se embeddea y se recupera como “documento”.
- `metadatas.append({...})` con:
  - `doc_id` (string)
  - `doc_nombre` (nombre del archivo PDF, truncado a 500 caracteres)
  - `pagina_inicio`, `pagina_fin` (enteros)
  - `hash_contenido`: primeros 16 hex de SHA-256 del texto UTF-8 (deduplicación / trazabilidad)
  - `capitulo`, `seccion`: reservados para futura extracción de estructura; hoy van vacíos.

Chroma exige que los valores de metadatos sean tipos permitidos (string, int, float, bool).

#### 6.5 Inserción en Chroma (`collection.add`)

Los chunks de ese PDF se suben en **lotes de 64** (`batch_size = 64`).  
Chroma, usando la `embedding_function` configurada, **calcula los vectores** para cada texto y los almacena junto a metadatos.

### Paso 7 — Resumen final

1. `collection.count()` devuelve el número total de fragmentos en la colección.
2. Se imprime la lista de **`doc_id`** únicos detectados para que puedas cruzarlos con el front y la API.

Código de salida **0** si todo fue bien.

---

## Qué archivos aparecen en `data/chroma`

Tras una ingestión exitosa verás típicamente:

- **`chroma.sqlite3`**: metadatos y gestión interna de Chroma.
- Carpetas con UUID (por ejemplo `9ff61fb2-.../`): archivos binarios del índice vectorial (HNSW u estructuras internas según versión de Chroma).

Esa carpeta **es** la “base de datos vectorizada” en modo local. Está en `.gitignore` para no subir gigabytes al repositorio.

---

## Cómo funciona la consulta (después de vectorizar)

Flujo en `app/vector_store.py` (retriever `ChromaRetriever`):

1. Se abre el mismo `PMDI_CHROMA_PATH` y la colección `pmdi_corpus` con la **misma** `SentenceTransformerEmbeddingFunction` y modelo que en ingestión (debe coincidir `PMDI_EMBED_MODEL`).
2. Opcionalmente se filtra por `doc_id` si el cliente envía ese campo.
3. Se generan **variantes de la pregunta** (`_query_variants`) para mejorar recall en temas como OKRs o alcance geográfico.
4. Chroma consulta con **`n_results` = al menos 10** por variante (aunque `top_k` sea 5) para tener margen antes del rerank.
5. Cada candidato obtiene `vector_score = clamp(1 - distancia, 0, 1)` (distancia coseno de Chroma). Los que tienen **`vector_score < 0.58`** se descartan antes del ranking final.
6. **Score combinado**: `0.78 * vector_score + 0.22 * lexical_overlap`, donde el solapamiento léxico usa tokens de la pregunta (sin stopwords cortas) frente al texto del fragmento.
7. **Boost opcional (+0.06**, acotado a 1.0): si la pregunta menciona Medellín/Colombia/alcance y el fragmento también menciona Medellín o Colombia.
8. Por cada `chunk_id` se conserva la mejor combinación entre todas las variantes de consulta; luego se ordena por score combinado y se descartan filas con **combinado < `MIN_SCORE` (0.65)**.
9. Se devuelven hasta `TOP_K` (5) `SourceChunk` al endpoint de chat.

La **respuesta en lenguaje natural** la genera `app/answering.py`: plantilla local (`PMDI_LLM_PROVIDER=local`) o Azure OpenAI si está configurado y disponible.

---

## Solución de problemas

| Síntoma | Qué revisar |
|---------|-------------|
| `Chroma no inicializado` o carpeta ausente | Ejecutar `python -m ingest.run_ingest` y verificar `PMDI_CHROMA_PATH`. |
| `Colección vacía` | Corpus sin texto extraíble o ingestión interrumpida; revisar logs por PDF. |
| Errores de socket al arrancar (Windows) | Probar otro puerto (`PMDI_API_PORT`) y `PMDI_API_RELOAD=0`. |
| Respuestas pobres o irrelevantes | Ajustar umbrales en `vector_store.py`, mejorar chunking, o activar LLM Azure; el plan técnico prevé también BM25 + fusión híbrida en una fase posterior. |
| Chat muy lento (varios segundos o más) | Ver sección **Rendimiento** abajo. En Render free tier, el **cold start** suma 30–60 s si el servicio estaba dormido. |
| Conflictos de dependencias al hacer `pip install` | Usar un **venv dedicado** solo para este proyecto. |

---

## Rendimiento del chat

Cada consulta embeddea la pregunta con **Sentence Transformers** (local) y busca en Chroma. La lentitud suele deberse a:

1. **Primera petición** tras arrancar el servidor: carga del modelo (~10–30 s). Mitigación: `PMDI_WARMUP_ON_START=1` (por defecto) precarga al iniciar la API.
2. **Varias variantes de consulta** por pregunta (más embeddings). Por defecto `PMDI_QUERY_VARIANTS=0` (una sola). Pon `1` si priorizas recall sobre velocidad.
3. **Render / Cloud Run en frío**: plan gratuito apaga el contenedor; la primera pregunta despierta el servicio.
4. **Azure OpenAI** en generación: añade latencia de red + tokens; la recuperación sigue siendo local salvo que migres embeddings a Azure.

Variables útiles en `.env`:

| Variable | Default | Efecto |
|----------|---------|--------|
| `PMDI_WARMUP_ON_START` | `1` | Carga índice + modelo al arrancar |
| `PMDI_QUERY_VARIANTS` | `0` | Desactiva consultas extra (más rápido) |
| `PMDI_RETRIEVAL_N_RESULTS` | `8` | Candidatos por búsqueda (menor = algo más rápido) |

Tras desplegar de nuevo el backend, prueba `/health` una vez (dispara warmup) y luego el chat.

---

## Próximos pasos (Azure, sin reescribir todo el front)

1. **LLM:** `PMDI_LLM_PROVIDER=azure_openai` + variables `AZURE_OPENAI_*`.
2. **Índice en la nube:** implementar carga y consulta en `PMDI_VECTOR_PROVIDER=azure_ai_search` y rellenar `AZURE_SEARCH_*`. Los PDFs y el mismo diseño de chunks pueden reutilizarse; el destino del índice cambia.

---

## Despliegue demo: Firebase Hosting (front) + API externa (back)

Los **embeddings** siguen en **Chroma** (`data/chroma/`). El índice va **dentro de la imagen Docker** del backend (generar con `python -m ingest.run_ingest` antes del build).

El front en Firebase **solo sirve archivos estáticos** (como otros proyectos Angular). El navegador llama al backend por **URL pública** configurada en `frontend/src/environments/environment.prod.ts` (`apiBaseUrl`).

### 1. Backend (Render, Railway, Cloud Run con URL propia, etc.)

1. Build Docker desde `backend/` con `data/chroma/` presente.
2. Despliega el contenedor en tu plataforma (ej. Render Web Service, puerto `8080` o variable `PORT`).
3. En el back, configura CORS para el dominio de Firebase:
   ```env
   PMDI_CORS_ORIGINS=https://innovacion-demo.web.app,https://innovacion-demo.firebaseapp.com
   ```
4. Copia la URL pública del servicio (ej. `https://pmdi-api.onrender.com`).

### 2. Frontend en Firebase Hosting

1. En `environment.prod.ts`, pon la misma URL en `apiBaseUrl` (sin barra final).
2. Desde `frontend/`:

```bash
npm run build
firebase deploy --only hosting --project innovacion-demo
```

Sitio esperado: `https://innovacion-demo.web.app`

### 3. Credenciales

El JSON **firebase-adminsdk** es para Admin SDK / CI, no para el Angular en el navegador. No lo subas al repositorio (está en `.gitignore`). Para CLI usa `firebase login`.

---

## Licencia y confidencialidad

El contenido del corpus PMDI puede estar sujeto a restricciones institucionales. No subas `data/chroma` ni PDFs a repositorios públicos sin autorización.
