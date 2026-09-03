from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.dependencies import init_services, get_rag_service, get_session_service
from app.routers import chat, documents
from app.schemas.chat import HealthResponse
from app.middleware.error_handler import global_exception_handler

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте, очистка при остановке."""
    settings = get_settings()
    rag_service, session_service = init_services(settings)

    # Пытаемся загрузить существующий индекс
    loaded = rag_service.load_index()
    if loaded:
        print(f"✅ Индекс загружен: {rag_service.vectorstore.index.ntotal} чанков")
    else:
        print("⚠️ Индекс не найден. Загрузите документы через /documents/upload")

    yield # приложение работает

    # Здесь можно добавить очистку ресурсов
    print("🛑 Приложение остановлено")


app = FastAPI(
    title="RAG Chatbot API",
    description="Чат-бот с поиском по документам на LangChain + FastAPI",
    version="1.0.0",
    lifespan=lifespan
)


# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_exception_handler(Exception, global_exception_handler)

# Подключаем роутеры
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    rag = get_rag_service()
    sessions = get_session_service()
    return HealthResponse(
        status="ok",
        index_loaded=rag.is_ready,
        active_sessions=sessions.active_count,
        timestamp=datetime.utcnow(),
    )