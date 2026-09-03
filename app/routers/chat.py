import time
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatResponse, ChatRequest
from app.services.rag import RAGService
from app.services.session import SessionService
from app.dependencies import get_rag_service, get_session_service
import json


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
    sessions: SessionService = Depends(get_session_service),
):
    """Обычный (не стриминговый) эндпоинт."""
    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="Индекс не загружен")

    history = sessions.get_or_create(request.session_id)
    start = time.time()

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                rag.chain_with_sources.invoke,
                {
                    "question": request.question,
                    "history": history.messages,
                }
            ),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Превышен таймаут запроса")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

    # Обновляем историю
    history.add_user_message(request.question)
    history.add_ai_message(result["answer"])

    return ChatResponse(
        session_id=request.session_id,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=round((time.time() - start) * 1000, 1),
    )


@router.get("/stream")
async def chat_stream(
    session_id: str,
    question: str,
    rag: RAGService = Depends(get_rag_service),
    sessions: SessionService = Depends(get_session_service),
):
    """Стриминговый эндпоинт через Server-Sent Events."""
    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="Индекс не загружен")

    history = sessions.get_or_create(session_id)

    async def event_generator():
        full_answer = ""
        try:
            # Сначала находим источники (быстро)
            sources = await asyncio.to_thread(
                lambda: [
                    d.metadata.get("source", f"doc_{i}")
                    for i, d in enumerate(rag.retriever.invoke(question))
                ]
            )

            # Отправляем источники как первое событие
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            for chunk in rag.chain.stream({
                "question": question,
                "history": history.messages,
            }):
                if chunk:
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'full_answer': full_answer})}\n\n"

            # Обновляем историю после стриминга

            history.add_user_message(question)
            history.add_ai_message(full_answer)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", # отключаем буферизацию в nginx
        },
    )

@router.delete("/{session_id}")
async def clear_session(
    session_id: str,
    sessions: SessionService = Depends(get_session_service),
):
    """Очищает историю сессии."""
    cleared = sessions.clear(session_id)
    return {"cleared": cleared, "session_id": session_id}