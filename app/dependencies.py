from app.config import Settings, get_settings
from app.services.rag import RAGService
from app.services.session import SessionService


# Синглтоны создаются один раз при старте
_rag_service: RAGService | None = None
_session_service: SessionService | None = None


def get_rag_service() -> RAGService:
    return _rag_service


def get_session_service() -> SessionService:
    return _session_service


def init_services(settings: Settings):
    global _rag_service, _session_service
    _rag_service = RAGService(settings)
    _session_service = SessionService(settings)
    return _rag_service, _session_service