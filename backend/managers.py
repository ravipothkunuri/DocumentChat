"""
Managers for metadata and models
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_ollama import OllamaEmbeddings

from backend.config import ConfigManager, METADATA_FILE, OLLAMA_BASE_URL
from backend.models import OllamaLLM

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manage document metadata"""
    
    def __init__(self, metadata_file: Path = METADATA_FILE):
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self) -> None:
        """Load metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}
    
    def save(self) -> None:
        """Save metadata to file"""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add(self, filename: str, metadata: Dict[str, Any]) -> None:
        """Add or update document metadata"""
        self.metadata[filename] = metadata
        self.save()
    
    def remove(self, filename: str) -> bool:
        """Remove document metadata"""
        if filename in self.metadata:
            del self.metadata[filename]
            self.save()
            return True
        return False
    
    def get(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        return self.metadata.get(filename)
    
    def exists(self, filename: str) -> bool:
        """Check if document exists"""
        return filename in self.metadata
    
    def clear(self) -> None:
        """Clear all metadata"""
        self.metadata = {}
        self.save()
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all document metadata"""
        return list(self.metadata.values())


class ModelManager:
    """Manage LLM and embedding models"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.embeddings_model: Optional[OllamaEmbeddings] = None
        self.llm_cache: Dict[str, OllamaLLM] = {}
    
    def get_embeddings_model(self) -> OllamaEmbeddings:
        """Get or create embeddings model"""
        if self.embeddings_model is None:
            model_name = self.config.get('embedding_model')
            logger.info(f"Initializing embeddings model: {model_name}")
            
            try:
                self.embeddings_model = OllamaEmbeddings(
                    model=model_name,
                    base_url=OLLAMA_BASE_URL
                )
                
                test_embedding = self.embeddings_model.embed_query("test")
                logger.info(f"Embeddings model '{model_name}' ready, dimensions: {len(test_embedding)}")
            except Exception as e:
                logger.error(f"Error initializing embeddings model '{model_name}': {e}")
                raise ValueError(f"Failed to initialize embeddings model: {str(e)}")
        
        return self.embeddings_model
    
    def get_llm_model(self, model_name: Optional[str] = None, temperature: Optional[float] = None) -> OllamaLLM:
        """Get or create LLM model"""
        model_to_use = model_name or self.config.get('model')
        
        if model_to_use not in self.llm_cache:
            logger.info(f"Initializing LLM model: {model_to_use}")
            
            try:
                llm = OllamaLLM(
                    model=model_to_use,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temperature or self.config.get('temperature'),
                    timeout=self.config.get('timeout', 120),
                    cold_start_timeout=self.config.get('cold_start_timeout', 600)
                )
                self.llm_cache[model_to_use] = llm
            except Exception as e:
                logger.error(f"Error initializing LLM model '{model_to_use}': {e}")
                raise ValueError(f"Failed to initialize LLM model: {str(e)}")
        
        if temperature is not None:
            self.llm_cache[model_to_use].temperature = temperature
        
        return self.llm_cache[model_to_use]
    
    def reset_embeddings_model(self) -> None:
        """Reset embeddings model"""
        self.embeddings_model = None
    
    def reset_llm_cache(self) -> None:
        """Reset LLM cache"""
        self.llm_cache.clear()
