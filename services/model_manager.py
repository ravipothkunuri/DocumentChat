"""Model manager responsible for lazy initialization of embeddings and LLM.

Centralizes access to the embeddings model and the async LLM client with
configurable temperature. Provides `cleanup` for graceful shutdown.
"""
from typing import Optional

from langchain_ollama import OllamaEmbeddings

from configuration import OLLAMA_BASE_URL, DEFAULT_MODEL
from services.llm import AsyncOllamaLLM
from stores.json_store import JSONStore


class ModelManager:
    def __init__(self, config: JSONStore):
        self.config = config
        self._embeddings_model: Optional[OllamaEmbeddings] = None
        self._llm: Optional[AsyncOllamaLLM] = None

    @property
    def embeddings_model(self) -> OllamaEmbeddings:
        if self._embeddings_model is None:
            self._embeddings_model = OllamaEmbeddings(
                model=self.config.get('embedding_model'),
                base_url=OLLAMA_BASE_URL,
            )
        return self._embeddings_model

    @property
    def llm(self) -> AsyncOllamaLLM:
        if self._llm is None:
            self._llm = AsyncOllamaLLM(
                model=DEFAULT_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=self.config.get('temperature', 0.7),
            )
        return self._llm

    async def cleanup(self) -> None:
        if self._llm:
            await self._llm.close()


