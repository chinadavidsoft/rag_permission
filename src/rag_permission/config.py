from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "changeme"
    llm_model: str = "qwen2.5:7b-instruct"
    qdrant_url: str = ""
    collection_name: str = "rag_permission_chunks"
    sqlite_path: Path = Path("./data/rag_permission.db")

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    hf_endpoint: str = "https://hf-mirror.com"
    hf_hub_disable_xet: bool = True
    enable_rerank: bool = True

    fixture_dir: Path = Path("fixtures")
    chunk_size: int = 260
    chunk_overlap: int = 48
    parent_chunk_size: int = 1000
    ingestion_strategy: str = "parent_child"
    dense_top_k: int = 24
    bm25_top_k: int = 24
    final_top_k: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
