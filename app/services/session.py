import time
from langchain_community.chat_message_histories import ChatMessageHistory
from app.config import Settings


class SessionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions: dict[str, dict] = {}

    def get_or_create(self, session_id: str) -> ChatMessageHistory:
        now = time.time()

        # Очищаем устаревшие сессии
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data["last_active"] > self.settings.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "history": ChatMessageHistory(),
                "last_active": now,
            }

        self._sessions[session_id]["last_active"] = now
        history = self._sessions[session_id]["history"]

        # Обрезаем историю до лимита
        if len(history.messages) > self.settings.max_history_messages:
            history.messages = history.messages[-self.settings.max_history_messages:]

        return history

    def clear(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    @property
    def active_count(self) -> int:
        return len(self._sessions)