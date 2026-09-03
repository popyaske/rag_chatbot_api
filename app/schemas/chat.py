from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID сессии пользователя")
    question: str = Field(..., min_length=1, max_length=2000)
    stream: bool = Field(default=True, description="Стриминг ответа")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str] = []
    tokens_used: int = 0
    latency_ms: float = 0.0


class DocumentUploadResponse(BaseModel):
    message: str
    chunks_indexed: int
    index_size: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    active_sessions: int
    timestamp: datetime