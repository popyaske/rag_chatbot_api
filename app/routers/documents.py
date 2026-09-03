import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.schemas.chat import DocumentUploadResponse
from app.services.rag import RAGService
from app.dependencies import get_rag_service


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag: RAGService = Depends(get_rag_service),
):
    """Загружает документ и добавляет его в индекс."""
    allowed_extensions = {".pdf", ".txt", ".md"}

    _, ext = os.path.splitext(file.filename or "")
    suffix = ext.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип файла. Разрешены: {', '.join(allowed_extensions)}",
        )

    # Сохраняем во временный файл
    with tempfile.TemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Загружаем документ
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path,  encoding="utf-8")

        docs = loader.load()

        # Добавляем метаданные источника
        for doc in docs:
            doc.metadata["source"] = file.filename

        chunks_count = rag.index_documents(docs)

        return DocumentUploadResponse(
            message=f"Файл '{file.filename}' проиндексирован",
            chunks_indexed=chunks_count,
            index_size=rag.vectorstore.index.ntotal,
        )
    finally:
        os.unlink(tmp_path) # удаляем временный файл


@router.get("/stats")
async def index_stats(rag: RAGService = Depends(get_rag_service)):
    """Статистика индекса."""
    if not rag.is_ready:
        return {"status": "not_loaded", "total_chunks": 0}
    return {
        "status": "ready",
        "total_chunks": rag.vectorstore.index.ntotal,
    }