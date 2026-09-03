import os
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from app.config import Settings


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = ChatOllama(
            model=settings.model_name,
            num_ctx=settings.max_tokens
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore: FAISS | None = None
        self.retriever: None = None
        self.chain = None
        self.chain_with_sources = None

    def load_index(self) -> bool:
        """Загружает FAISS-индекс с диска. Возвращает True если успешно."""
        if not os.path.exists(self.settings.index_path):
            return False
        try:
            self.vectorstore = FAISS.load_local(
                self.settings.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._build_chain()
            return True
        except Exception as e:
            print(f"Ошибка загрузки индекса: {e}")
            return False

    def index_documents(self, docs: list[Document]) -> int:
        """Добавляет документы в индекс. Возвращает количество чанков."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        os.makedirs(os.path.dirname(self.settings.index_path), exist_ok=True)
        self.vectorstore.save_local(self.settings.index_path)
        self._build_chain()
        return len(chunks)

    def _build_chain(self):
        """Собирает RAG-цепочку после загрузки/обновления индекса."""
        # Гибридный retriever
        all_docs = list(self.vectorstore.docstore._dict.values())
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = self.settings.retriever_k

        vector = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.settings.retriever_k, "fetch_k": 20},
        )

        self.retriever = EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=[0.3, 0.7]
        )


        # Промпт с историей
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты технический ассистент. Отвечай только на основе контекста.
        Если ответа нет в контексте — скажи: "В базе знаний нет информации по этому вопросу."
        Не выдумывай факты.

        Контекст:
        {context}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(f"Источник {i+1}] {d.page_content}" for i, d in enumerate(docs))

        self.chain = (
            RunnablePassthrough.assign(
                context=lambda x:format_docs(self.retriever.invoke(x["question"]))
            )
            | prompt
            | self.model
            | StrOutputParser()
        )

        self.chain_with_sources = RunnableParallel(
            answer=self.chain,
            sources=lambda x: list(set(
                d.metadata.get("source", f"doc_{i}")
                for i, d in enumerate(self.retriever.invoke(x["question"])))
            )
        )

    @property
    def is_ready(self) -> bool:
        return self.chain is not None