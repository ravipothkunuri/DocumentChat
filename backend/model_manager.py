"""Model Manager - 35 lines"""
import logging
from typing import Optional
from langchain_ollama import OllamaEmbeddings
from backend.config import OLLAMA_BASE_URL, FIXED_MODEL
from backend.ollama import AsyncOllamaLLM
from backend.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage LLM and embedding models"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.embeddings_model: Optional[OllamaEmbeddings] = None
        self.llm: Optional[AsyncOllamaLLM] = None
    
    def get_embeddings_model(self) -> OllamaEmbeddings:
        """Get or create embeddings model"""
        if self.embeddings_model is None:
            model_name = self.config.get('embedding_model')
            logger.info(f"Initializing embeddings: {model_name}")
            self.embeddings_model = OllamaEmbeddings(
                model=model_name, 
                base_url=OLLAMA_BASE_URL
            )
        return self.embeddings_model
    
    def get_llm(self) -> AsyncOllamaLLM:
        """Get or create LLM model"""
        if self.llm is None:
            logger.info(f"Initializing LLM: {FIXED_MODEL}")
            self.llm = AsyncOllamaLLM(
                model=FIXED_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=self.config.get('temperature', 0.7)
            )
        return self.llm
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.llm:
            await self.llm.close()
