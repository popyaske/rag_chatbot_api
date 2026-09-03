from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    model_name: str = "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"

    #LangSmith (опционально)
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "rag-chatbot"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4
    index_path: str = "indexes/faiss_index"

    # Сессии
    max_history_messages: int = 20
    session_ttl_seconds: int = 3600

    # API
    max_tokens: int = 8000
    request_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()