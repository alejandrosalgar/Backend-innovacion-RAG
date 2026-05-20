from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = Field(default="127.0.0.1", alias="PMDI_API_HOST")
    api_port: int = Field(default=8080, alias="PMDI_API_PORT")
    api_reload: bool = Field(default=False, alias="PMDI_API_RELOAD")

    # Orígenes extra para CORS (coma-separados), p. ej. Firebase Hosting en producción
    cors_origins_extra: str = Field(default="", alias="PMDI_CORS_ORIGINS")

    vector_provider: str = Field(default="chroma", alias="PMDI_VECTOR_PROVIDER")
    llm_provider: str = Field(default="local", alias="PMDI_LLM_PROVIDER")

    # Local / Chroma
    chroma_path: str = Field(default="./data/chroma", alias="PMDI_CHROMA_PATH")
    embed_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="PMDI_EMBED_MODEL",
    )
    # Cargar Chroma + modelo de embeddings al arrancar (evita 10–30 s en la 1.ª pregunta).
    warmup_on_start: bool = Field(default=True, alias="PMDI_WARMUP_ON_START")
    # Variantes extra de consulta mejoran recall pero duplican embedding por request.
    query_variants_enabled: bool = Field(default=False, alias="PMDI_QUERY_VARIANTS")
    retrieval_n_results: int = Field(default=8, alias="PMDI_RETRIEVAL_N_RESULTS")
    min_retrieval_score: float = Field(default=0.58, alias="PMDI_MIN_RETRIEVAL_SCORE")
    min_vector_score: float = Field(default=0.48, alias="PMDI_MIN_VECTOR_SCORE")
    retrieval_fallback: bool = Field(default=True, alias="PMDI_RETRIEVAL_FALLBACK")

    # Azure OpenAI
    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    )

    # Azure AI Search (reservado para siguiente paso)
    azure_search_endpoint: str | None = Field(default=None, alias="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str | None = Field(default=None, alias="AZURE_SEARCH_API_KEY")
    azure_search_index: str | None = Field(default=None, alias="AZURE_SEARCH_INDEX")

    @property
    def chroma_abs_path(self) -> Path:
        return Path(self.chroma_path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()

