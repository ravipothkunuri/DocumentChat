"""
Manager Classes for the RAG Application

This file brings together three important managers that keep our app running smoothly:
- ConfigManager: Remembers your settings between sessions
- MetadataManager: Keeps track of all uploaded documents
- ModelManager: Handles our AI models efficiently

Think of these as the "memory" of our application!
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from langchain_ollama import OllamaEmbeddings
from backend.config import (
    CONFIG_FILE, METADATA_FILE, OLLAMA_BASE_URL, 
    FIXED_MODEL, DEFAULT_EMBEDDING_MODEL
)
from backend.ollama import AsyncOllamaLLM

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """
    Saves and loads your app settings automatically.
    
    This manager keeps track of things like:
    - Which AI model you're using
    - How big your document chunks should be
    - How many questions you've asked (for stats!)
    
    Everything gets saved to a JSON file, so your settings survive restarts.
    Pretty neat, right?
    """
    
    # These are our sensible defaults
    DEFAULT_CONFIG = {
        'model': FIXED_MODEL,
        'embedding_model': DEFAULT_EMBEDDING_MODEL,
        'chunk_size': 1000,              # Sweet spot for most documents
        'chunk_overlap': 200,             # Keeps context between chunks
        'temperature': 0.7,               # Not too random, not too boring
        'total_queries': 0
    }
    
    def __init__(self, config_file: Path = CONFIG_FILE):
        """Set up the config manager and load existing settings if available."""
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """
        Try to load settings from disk.
        
        If there's no config file yet, we'll create one with defaults.
        If something goes wrong, we log it and keep using defaults.
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
                logger.info(f"Loaded config from {self.config_file}")
            else:
                self.save()  # Create a new config file
                logger.info("Created fresh config file with defaults")
        except Exception as e:
            logger.error(f"Couldn't load config: {e}")
            # No worries though, we'll just use defaults
    
    def save(self):
        """
        Save current settings to disk.
        
        We make sure the directory exists first, then write everything
        as pretty-printed JSON so it's human-readable.
        """
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug(f"Saved config to {self.config_file}")
        except Exception as e:
            logger.error(f"Couldn't save config: {e}")
    
    def get(self, key: str, default=None):
        """
        Get a setting value.
        
        Usage:
            chunk_size = config.get('chunk_size')
            temp = config.get('temperature', 0.7)  # with fallback
        """
        return self.config.get(key, default)
    
    def increment_queries(self):
        """
        Bump up the query counter by one.
        
        This helps us track how much the app is being used. We automatically
        save after incrementing so the count survives restarts.
        """
        self.config['total_queries'] += 1
        self.save()
        logger.debug(f"Query count now at {self.config['total_queries']}")


# ============================================================================
# METADATA MANAGER
# ============================================================================

class MetadataManager:
    """
    Keeps track of all your uploaded documents.
    
    For each document, we remember stuff like:
    - When you uploaded it
    - How big it is
    - How many chunks we split it into
    - What type of file it was
    
    All this info gets saved to JSON, so we don't lose track of anything!
    """
    
    def __init__(self, metadata_file: Path = METADATA_FILE):
        """Initialize and load any existing document metadata."""
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self):
        """
        Load document info from disk.
        
        If there's no metadata file yet, that's fine - we'll start fresh!
        """
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Found metadata for {len(self.metadata)} documents")
        except Exception as e:
            logger.error(f"Couldn't load metadata: {e}")
            self.metadata = {}  # Start with clean slate
    
    def save(self):
        """
        Save all document info to disk.
        
        We use pretty JSON so you can actually read it if you're curious!
        """
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Saved metadata for {len(self.metadata)} documents")
        except Exception as e:
            logger.error(f"Couldn't save metadata: {e}")
    
    def add(self, filename: str, metadata: Dict[str, Any]):
        """
        Add or update info for a document.
        
        Pass in the filename and a dictionary with all the details.
        We'll save it automatically.
        """
        self.metadata[filename] = metadata
        self.save()
        logger.info(f"Registered {filename} in metadata")
    
    def remove(self, filename: str) -> bool:
        """
        Delete a document's metadata.
        
        Returns True if we found and removed it, False if it wasn't there.
        """
        if filename in self.metadata:
            del self.metadata[filename]
            self.save()
            logger.info(f"Removed {filename} from metadata")
            return True
        logger.warning(f"Tried to remove {filename} but it wasn't in metadata")
        return False
    
    def get(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get info about a specific document. Returns None if not found."""
        return self.metadata.get(filename)
    
    def exists(self, filename: str) -> bool:
        """Quick check: do we have this document?"""
        return filename in self.metadata
    
    def list_all(self) -> List[Dict[str, Any]]:
        """Get info about all documents as a list."""
        return list(self.metadata.values())


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Handles our AI models smartly.
    
    We use "lazy loading" here - models only get initialized when you actually
    need them. This saves memory and startup time!
    
    Once a model is loaded, we keep it around (like a cache) so we don't have
    to reload it every time.
    """
    
    def __init__(self, config: ConfigManager):
        """Set up the model manager with our config."""
        self.config = config
        self.embeddings_model: Optional[OllamaEmbeddings] = None
        self.llm: Optional[AsyncOllamaLLM] = None
    
    def get_embeddings_model(self) -> OllamaEmbeddings:
        """
        Get the embedding model (for turning text into vectors).
        
        First call: We load the model
        Subsequent calls: We return the cached model
        
        This pattern saves a TON of time!
        """
        if self.embeddings_model is None:
            model_name = self.config.get('embedding_model')
            logger.info(f"Loading embeddings model: {model_name}")
            self.embeddings_model = OllamaEmbeddings(
                model=model_name, 
                base_url=OLLAMA_BASE_URL
            )
        return self.embeddings_model
    
    def get_llm(self) -> AsyncOllamaLLM:
        """
        Get the language model (for generating answers).
        
        Same lazy loading pattern as embeddings - we only create it once!
        """
        if self.llm is None:
            logger.info(f"Loading LLM: {FIXED_MODEL}")
            self.llm = AsyncOllamaLLM(
                model=FIXED_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=self.config.get('temperature', 0.7)
            )
        return self.llm
    
    async def cleanup(self):
        """
        Clean up when shutting down.
        
        This properly closes connections and prevents memory leaks.
        Always call this when the app is stopping!
        """
        if self.llm:
            await self.llm.close()
            logger.info("Cleaned up LLM resources")
