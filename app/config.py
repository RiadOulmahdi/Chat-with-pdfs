from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    pdf_source_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    chroma_collection_name: str = "pdf_rag"
    ingestion_state_db: Path = PROJECT_ROOT / "data" / "db" / "ingestion_state.db"
    metrics_log_path: Path = PROJECT_ROOT / "data" / "metrics" / "query_metrics.jsonl"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 5

    embedding_batch_size: int = 128
    ingestion_workers: int = 8


settings = Settings()
