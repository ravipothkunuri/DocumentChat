import os
import json
import logging
import traceback
import subprocess
import re
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Iterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain_ollama import OllamaEmbeddings

from vector_store import VectorStore, VectorStoreError

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_BASE_URL = "http://localhost:11434"
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
METADATA_FILE = VECTOR_DIR / "metadata.json"
CONFIG_FILE = VECTOR_DIR / "config.json"
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Pydantic Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="The question to ask")
    model: Optional[str] = Field(None, description="LLM model to use")
    top_k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature")
    stream: bool = Field(False, description="Stream response in real-time")


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: int
    similarity_scores: List[float]
    processing_time: float


class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str


class ModelConfig(BaseModel):
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = Field(None, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if v is not None and 'chunk_size' in values and values['chunk_size'] is not None:
            if v >= values['chunk_size']:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class DocumentInfo(BaseModel):
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str


# Universal Ollama LLM with automatic endpoint detection
class UniversalOllamaLLM:
    """
    Universal Ollama LLM client with automatic endpoint detection.
    Supports both /api/generate and /api/chat endpoints.
    """
    
    def __init__(
        self, 
        model: str, 
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        timeout: int = 120
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.timeout = timeout
        self.endpoint_type = None  # Will be detected automatically
        
        logger.info(f"Initializing UniversalOllamaLLM with model: {model}")
        self._detect_endpoint()
    
    def _detect_endpoint(self) -> None:
        """Automatically detect which endpoint the model supports."""
        try:
            # Try chat endpoint first (preferred for conversational models)
            chat_url = f"{self.base_url}/api/chat"
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False
            }
            
            response = requests.post(
                chat_url, 
                json=test_payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                self.endpoint_type = "chat"
                logger.info(f"Model {self.model} supports /api/chat endpoint")
                return
        except Exception as e:
            logger.debug(f"Chat endpoint test failed: {e}")
        
        try:
            # Try generate endpoint
            generate_url = f"{self.base_url}/api/generate"
            test_payload = {
                "model": self.model,
                "prompt": "test",
                "stream": False
            }
            
            response = requests.post(
                generate_url, 
                json=test_payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                self.endpoint_type = "generate"
                logger.info(f"Model {self.model} supports /api/generate endpoint")
                return
        except Exception as e:
            logger.debug(f"Generate endpoint test failed: {e}")
        
        # Default to generate if detection fails
        self.endpoint_type = "generate"
        logger.warning(
            f"Could not detect endpoint for {self.model}, defaulting to /api/generate"
        )
    
    def _call_chat_endpoint(
        self, 
        prompt: str, 
        stream: bool = False
    ) -> requests.Response:
        """Call the /api/chat endpoint."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": stream,
            "options": {
                "temperature": self.temperature
            }
        }
        
        return requests.post(
            url, 
            json=payload, 
            timeout=self.timeout,
            stream=stream
        )
    
    def _call_generate_endpoint(
        self, 
        prompt: str, 
        stream: bool = False
    ) -> requests.Response:
        """Call the /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature
            }
        }
        
        return requests.post(
            url, 
            json=payload, 
            timeout=self.timeout,
            stream=stream
        )
    
    def invoke(self, prompt: str) -> str:
        """
        Invoke the model with a prompt and return the complete response.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The model's response as a string
        """
        try:
            if self.endpoint_type == "chat":
                response = self._call_chat_endpoint(prompt, stream=False)
            else:
                response = self._call_generate_endpoint(prompt, stream=False)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response based on endpoint type
            if self.endpoint_type == "chat":
                return data.get("message", {}).get("content", "")
            else:
                return data.get("response", "")
        
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for model {self.model}")
            raise HTTPException(
                status_code=504,
                detail=f"Model {self.model} request timed out"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for model {self.model}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to communicate with model {self.model}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error invoking model {self.model}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
    
    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the model's response.
        
        Args:
            prompt: The input prompt
            
        Yields:
            Response chunks as strings
        """
        try:
            if self.endpoint_type == "chat":
                response = self._call_chat_endpoint(prompt, stream=True)
            else:
                response = self._call_generate_endpoint(prompt, stream=True)
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Extract content based on endpoint type
                        if self.endpoint_type == "chat":
                            content = data.get("message", {}).get("content", "")
                        else:
                            content = data.get("response", "")
                        
                        if content:
                            yield content
                        
                        # Check if done
                        if data.get("done", False):
                            break
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON: {line}")
                        continue
        
        except requests.exceptions.Timeout:
            logger.error(f"Streaming timeout for model {self.model}")
            raise HTTPException(
                status_code=504,
                detail=f"Model {self.model} streaming timed out"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming failed for model {self.model}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stream from model {self.model}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error streaming from model {self.model}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected streaming error: {str(e)}"
            )


# Configuration Manager
class ConfigManager:
    """Centralized configuration management."""
    
    DEFAULT_CONFIG = {
        'model': 'phi3',
        'embedding_model': 'nomic-embed-text',
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'temperature': 0.7,
        'total_queries': 0
    }
    
    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                self.config.update(loaded_config)
                logger.info(
                    f"Loaded configuration: model={self.config['model']}, "
                    f"embedding={self.config['embedding_model']}"
                )
            else:
                logger.info("No existing config file, using defaults")
                self.save()  # Save defaults
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update(self, **kwargs) -> List[str]:
        """Update configuration and return list of changed fields."""
        changed = []
        for key, value in kwargs.items():
            if value is not None and key in self.config:
                if self.config[key] != value:
                    self.config[key] = value
                    changed.append(key)
        
        if changed:
            self.save()
        
        return changed
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def increment_queries(self) -> None:
        """Increment query counter."""
        self.config['total_queries'] += 1
        self.save()


# Metadata Manager
class MetadataManager:
    """Manage document metadata."""
    
    def __init__(self, metadata_file: Path = METADATA_FILE):
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self) -> None:
        """Load metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} documents")
            else:
                self.metadata = {}
                logger.info("No existing metadata file found")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}
    
    def save(self) -> None:
        """Save metadata to file."""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add(self, filename: str, metadata: Dict[str, Any]) -> None:
        """Add or update document metadata."""
        self.metadata[filename] = metadata
        self.save()
    
    def remove(self, filename: str) -> bool:
        """Remove document metadata."""
        if filename in self.metadata:
            del self.metadata[filename]
            self.save()
            return True
        return False
    
    def get(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document metadata."""
        return self.metadata.get(filename)
    
    def exists(self, filename: str) -> bool:
        """Check if document exists."""
        return filename in self.metadata
    
    def clear(self) -> None:
        """Clear all metadata."""
        self.metadata = {}
        self.save()
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all document metadata."""
        return list(self.metadata.values())


# Model Manager
class ModelManager:
    """Manage LLM and embedding models."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.embeddings_model: Optional[OllamaEmbeddings] = None
        self.llm_model: Optional[UniversalOllamaLLM] = None
        self.llm_model_name: Optional[str] = None
    
    def get_embeddings_model(self) -> OllamaEmbeddings:
        """Get or create embeddings model."""
        try:
            if self.embeddings_model is None:
                model_name = self.config.get('embedding_model')
                logger.info(f"Initializing embeddings model: {model_name}")
                
                self.embeddings_model = OllamaEmbeddings(
                    model=model_name,
                    base_url=OLLAMA_BASE_URL
                )
                
                # Test the model
                test_embedding = self.embeddings_model.embed_query("test")
                logger.info(
                    f"Embeddings model '{model_name}' initialized successfully, "
                    f"dimensions: {len(test_embedding)}"
                )
            
            return self.embeddings_model
        
        except Exception as e:
            model_name = self.config.get('embedding_model')
            logger.error(f"Error initializing embeddings model '{model_name}': {e}")
            
            if "404" in str(e) or "not found" in str(e).lower():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Embedding model '{model_name}' not found. "
                        f"Pull it using: ollama pull {model_name}"
                    )
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize embeddings model '{model_name}': {str(e)}"
            )
    
    def get_llm_model(self, model_name: Optional[str] = None) -> UniversalOllamaLLM:
        """Get or create LLM model."""
        try:
            model_to_use = model_name or self.config.get('model')
            
            # Reinitialize if model changed
            if self.llm_model is None or self.llm_model_name != model_to_use:
                logger.info(f"Initializing LLM model: {model_to_use}")
                
                self.llm_model = UniversalOllamaLLM(
                    model=model_to_use,
                    temperature=self.config.get('temperature'),
                    base_url=OLLAMA_BASE_URL
                )
                self.llm_model_name = model_to_use
                logger.info(
                    f"LLM model '{model_to_use}' initialized successfully "
                    f"(endpoint: {self.llm_model.endpoint_type})"
                )
            
            return self.llm_model
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error initializing LLM model '{model_to_use}': {e}")
            
            if "404" in str(e) or "not found" in str(e).lower():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Model '{model_to_use}' not found. "
                        f"Pull it using: ollama pull {model_to_use}"
                    )
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize LLM model '{model_to_use}': {str(e)}"
            )
    
    def reset_embeddings_model(self) -> None:
        """Reset embeddings model (force reinitialization)."""
        self.embeddings_model = None
        logger.debug("Embeddings model reset")
    
    def reset_llm_model(self) -> None:
        """Reset LLM model (force reinitialization)."""
        self.llm_model = None
        self.llm_model_name = None
        logger.debug("LLM model reset")
    
    def update_temperature(self, temperature: float) -> None:
        """Update LLM temperature."""
        if self.llm_model:
            self.llm_model.temperature = temperature


# Document Processor
class DocumentProcessor:
    """Handle document loading and processing."""
    
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': UnstructuredWordDocumentLoader
    }
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def load_document(self, file_path: Path) -> List[str]:
        """Load document content based on file type."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix not in self.LOADERS:
                raise ValueError(f"Unsupported file type: {suffix}")
            
            loader_class = self.LOADERS[suffix]
            loader = loader_class(str(file_path))
            documents = loader.load()
            content = [doc.page_content for doc in documents]
            
            logger.debug(f"Loaded {len(content)} pages from {file_path}")
            return content
        
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        """Split text content into chunks."""
        try:
            full_text = "\n\n".join(content)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get('chunk_size'),
                chunk_overlap=self.config.get('chunk_overlap'),
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(full_text)
            
            documents = [
                {
                    "page_content": chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_id": i,
                        "chunk_length": len(chunk)
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
            
            logger.info(f"Split {filename} into {len(documents)} chunks")
            return documents
        
        except Exception as e:
            logger.error(f"Error splitting text for {filename}: {e}")
            raise


# Utility Functions
def clean_llm_response(text: str) -> str:
    """Clean LLM response by removing reasoning tags."""
    # Remove reasoning tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove standalone tags
    text = re.sub(r'</?(?:think|reasoning|thought)>', '', text, flags=re.IGNORECASE)
    
    # Clean whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_available_models() -> tuple[List[str], List[str]]:
    """Get available models from Ollama using direct API."""
    try:
        # Try using Ollama API directly
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'].split(':')[0] for model in data.get('models', [])]
            
            embedding_models = [
                m for m in models 
                if 'embed' in m.lower() or 'nomic' in m.lower()
            ]
            llm_models = [m for m in models if 'embed' not in m.lower()]
            
            return llm_models, embedding_models
    
    except Exception as e:
        logger.debug(f"API-based model detection failed: {e}")
    
    try:
        # Fallback to CLI
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0].split(':')[0]
                    models.append(model_name)
            
            embedding_models = [
                m for m in models 
                if 'embed' in m.lower() or 'nomic' in m.lower()
            ]
            llm_models = [m for m in models if 'embed' not in m.lower()]
            
            return llm_models, embedding_models
        
        logger.warning(f"Ollama list command failed: {result.stderr}")
        return [], []
    
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return [], []


def check_ollama_available(config: ConfigManager) -> tuple[bool, str]:
    """Check if Ollama is available using direct API calls."""
    try:
        # Check if Ollama server is running
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )
        
        if response.status_code != 200:
            return False, "Ollama server not responding"
        
        # Test embedding model
        embeddings = OllamaEmbeddings(
            model=config.get('embedding_model'),
            base_url=OLLAMA_BASE_URL
        )
        embeddings.embed_query("test")
        
        # Test LLM model using direct API
        llm_model = config.get('model')
        test_response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": llm_model,
                "prompt": "test",
                "stream": False
            },
            timeout=10
        )
        
        if test_response.status_code != 200:
            return False, f"LLM model {llm_model} not accessible"
        
        return True, f"Ollama available with {config.get('embedding_model')} and {llm_model}"
    
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, str(e)


def validate_file(filename: str, file_size: int) -> None:
    """Validate uploaded file.
    
    Raises:
        HTTPException: If validation fails
    """
    # Check file type
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File size ({file_size / (1024*1024):.1f} MB) exceeds "
                f"maximum allowed size of {MAX_FILE_SIZE_MB} MB"
            )
        )


# Initialize global objects
config_manager = ConfigManager()
metadata_manager = MetadataManager()
vector_store = VectorStore()
model_manager = ModelManager(config_manager)
document_processor = DocumentProcessor(config_manager)

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting RAG Application...")
    
    try:
        vector_store.load()
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load vector store: {e}")
    
    logger.info("RAG Application started successfully")
    
    yield
    
    logger.info("Shutting down RAG Application...")


# Initialize FastAPI app
app = FastAPI(
    title="LangChain Ollama RAG Assistant API",
    description="Production-ready RAG system with Universal Ollama LLM and custom vector store",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.get("/health", tags=["Health"])
async def health_check():
    """Check system health and configuration."""
    try:
        ollama_available, ollama_message = check_ollama_available(config_manager)
        stats = vector_store.get_stats()
        
        return {
            "status": "healthy" if ollama_available else "degraded",
            "timestamp": datetime.now().isoformat(),
            "ollama_status": {
                "available": ollama_available,
                "message": ollama_message
            },
            "configuration": {
                "model": config_manager.get('model'),
                "embedding_model": config_manager.get('embedding_model'),
                "chunk_size": config_manager.get('chunk_size'),
                "chunk_overlap": config_manager.get('chunk_overlap'),
                "temperature": config_manager.get('temperature')
            },
            "document_count": len(metadata_manager.metadata),
            "total_chunks": stats.get("total_chunks", 0),
            "total_queries": config_manager.get('total_queries')
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document synchronously."""
    logger.info(f"Upload request for file: {file.filename}")
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Validate file
    validate_file(file.filename, file_size)
    
    # Check for duplicates
    if metadata_manager.exists(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists. Delete it first or rename your file."
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Process document
        document_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(document_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        embeddings_model = model_manager.get_embeddings_model()
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        # Add to vector store
        vector_store.add_documents(documents, embeddings)
        
        # Save metadata
        metadata_manager.add(file.filename, {
            "filename": file.filename,
            "size": file_size,
            "chunks": len(documents),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        })
        
        vector_store.save()
        
        logger.info(
            f"Successfully processed {file.filename}: "
            f"{len(documents)} chunks, {file_size} bytes"
        )
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=file_size,
            message=f"Document processed successfully into {len(documents)} chunks"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        # Cleanup
        if file_path.exists():
            file_path.unlink()
        metadata_manager.remove(file.filename)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )


async def process_document_background(filename: str, file_path: Path, file_size: int):
    """Process document in background."""
    try:
        logger.info(f"Background processing started for {filename}")
        
        # Update status
        metadata = metadata_manager.get(filename)
        if metadata:
            metadata["status"] = "processing"
            metadata_manager.add(filename, metadata)
        
        # Process document
        document_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(document_content, filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        embeddings_model = model_manager.get_embeddings_model()
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        # Add to vector store
        vector_store.add_documents(documents, embeddings)
        
        # Update metadata
        metadata = metadata_manager.get(filename) or {}
        metadata.update({
            "chunks": len(documents),
            "status": "processed"
        })
        metadata_manager.add(filename, metadata)
        
        vector_store.save()
        
        logger.info(
            f"Background processing completed for {filename}: {len(documents)} chunks"
        )
    
    except Exception as e:
        logger.error(f"Error in background processing for {filename}: {e}")
        logger.debug(traceback.format_exc())
        
        # Update error status
        metadata = metadata_manager.get(filename) or {}
        metadata.update({
            "status": "failed",
            "error": str(e)
        })
        metadata_manager.add(filename, metadata)
        
        # Cleanup
        if file_path.exists():
            file_path.unlink()


@app.post("/upload/async", tags=["Documents"])
async def upload_document_async(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None
):
    """Upload a document and process it in the background."""
    logger.info(f"Async upload request for file: {file.filename}")
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Validate file
    validate_file(file.filename, file_size)
    
    # Check for duplicates
    if metadata_manager.exists(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists. Delete it first or rename your file."
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Save initial metadata
        metadata_manager.add(file.filename, {
            "filename": file.filename,
            "size": file_size,
            "chunks": 0,
            "status": "pending",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        })
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background, 
            file.filename, 
            file_path, 
            file_size
        )
        
        logger.info(
            f"File {file.filename} uploaded successfully, processing in background"
        )
        
        return {
            "status": "pending",
            "filename": file.filename,
            "file_size": file_size,
            "message": "Document uploaded successfully. Processing in background..."
        }
    
    except Exception as e:
        logger.error(f"Error during async upload {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        # Cleanup
        if file_path.exists():
            file_path.unlink()
        metadata_manager.remove(file.filename)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )


@app.post("/query", tags=["Query"])
async def query_documents(request: QueryRequest):
    """Query the document collection."""
    logger.info(
        f"Query request: '{request.question[:50]}...' "
        f"with model {request.model or config_manager.get('model')}, stream={request.stream}"
    )
    
    start_time = datetime.now()
    
    try:
        # Check if documents exist
        stats = vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents available. Please upload some documents first."
            )
        
        # Generate query embedding
        embeddings_model = model_manager.get_embeddings_model()
        query_embedding = embeddings_model.embed_query(request.question)
        
        # Perform similarity search
        similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
        if not similar_docs:
            raise HTTPException(
                status_code=400,
                detail="No relevant documents found for your query."
            )
        
        # Prepare context
        context_parts = []
        sources = []
        similarity_scores = []
        
        for doc, score in similar_docs:
            context_parts.append(
                f"Source: {doc['metadata']['source']}\n"
                f"Content: {doc['page_content']}"
            )
            sources.append(doc['metadata']['source'])
            similarity_scores.append(float(score))
        
        context = "\n\n".join(context_parts)
        
        # Get LLM
        llm = model_manager.get_llm_model(request.model)
        
        # Update temperature if specified
        if request.temperature is not None:
            llm.temperature = request.temperature
        
        # Build prompt
        unique_sources = list(set(sources))
        doc_identity = (
            unique_sources[0] if len(unique_sources) == 1
            else f"your documents ({', '.join(unique_sources[:2])}"
                 f"{'...' if len(unique_sources) > 2 else ''})"
        )
        
        prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person as if you are the document itself.

Your content includes:
{context}

The user asks: {request.question}

Respond naturally as the document, using "I" when referring to your content. For example:
- "Based on what I contain, I can tell you that..."
- "In my section about X, I mention that..."
- "I don't seem to have information about that topic."

Your response:"""
        
        # Stream or regular response
        if request.stream:
            async def generate() -> AsyncGenerator[str, None]:
                try:
                    # Send metadata first
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": similarity_scores,
                        "type": "metadata"
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"
                    
                    # Stream content
                    full_answer = ""
                    for chunk in llm.stream(prompt):
                        full_answer += chunk
                        # Send small chunks for smoother streaming
                        if len(chunk) > 0:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                    
                    # Clean and send any remaining content
                    cleaned_answer = clean_llm_response(full_answer)
                    
                    # Send completion
                    processing_time = (datetime.now() - start_time).total_seconds()
                    completion = {
                        "type": "done",
                        "processing_time": processing_time
                    }
                    yield f"data: {json.dumps(completion)}\n\n"
                    
                    config_manager.increment_queries()
                    logger.info(
                        f"Streaming query completed in {processing_time:.2f}s, "
                        f"{len(similar_docs)} chunks used"
                    )
                
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    error_msg = {
                        "type": "error",
                        "error": str(e)
                    }
                    yield f"data: {json.dumps(error_msg)}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        else:
            # Regular response
            answer = llm.invoke(prompt)
            answer = clean_llm_response(answer)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            config_manager.increment_queries()
            
            logger.info(
                f"Query completed in {processing_time:.2f}s, "
                f"{len(similar_docs)} chunks used"
            )
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                chunks_used=len(similar_docs),
                similarity_scores=similarity_scores,
                processing_time=processing_time
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all uploaded documents."""
    logger.debug("Listing documents")
    documents = [DocumentInfo(**meta) for meta in metadata_manager.list_all()]
    logger.info(f"Listed {len(documents)} documents")
    return documents


@app.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a specific document."""
    logger.info(f"Delete request for document: {filename}")
    
    if not metadata_manager.exists(filename):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found"
        )
    
    try:
        # Remove from vector store
        vector_store.remove_documents_by_source(filename)
        
        # Remove metadata
        metadata_manager.remove(filename)
        
        # Remove file
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        vector_store.save()
        
        logger.info(f"Successfully deleted document: {filename}")
        return {
            "message": f"Document '{filename}' deleted successfully",
            "note": "Vector store has been updated automatically"
        }
    
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.delete("/clear", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents and embeddings."""
    logger.info("Clear all documents request")
    
    try:
        vector_store.clear()
        metadata_manager.clear()
        
        # Remove all files
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        vector_store.save()
        
        logger.info("Successfully cleared all documents")
        return {
            "message": "All documents and embeddings cleared successfully",
            "cleared": True
        }
    
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear documents: {str(e)}"
        )


@app.get("/documents/{filename}/preview", tags=["Documents"])
async def preview_document(filename: str, num_chunks: int = Query(3, ge=1, le=10)):
    """Preview document chunks."""
    logger.debug(f"Preview request for {filename}, {num_chunks} chunks")
    
    if not metadata_manager.exists(filename):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found"
        )
    
    try:
        documents = vector_store.get_documents_by_source(filename)
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for document '{filename}'"
            )
        
        preview_chunks = documents[:num_chunks]
        
        chunks_data = [
            {
                "chunk_id": doc['metadata']['chunk_id'],
                "content": (
                    doc['page_content'][:500] + "..." 
                    if len(doc['page_content']) > 500 
                    else doc['page_content']
                ),
                "length": len(doc['page_content'])
            }
            for doc in preview_chunks
        ]
        
        logger.debug(f"Preview generated for {filename}: {len(chunks_data)} chunks")
        
        return {
            "filename": filename,
            "total_chunks": len(documents),
            "preview_chunks": len(chunks_data),
            "chunks": chunks_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating preview for {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )


@app.post("/configure", tags=["Configuration"])
async def configure_system(config_update: ModelConfig):
    """Update system configuration."""
    logger.info(f"Configuration update request: {config_update.dict(exclude_none=True)}")
    
    try:
        changed_fields = config_manager.update(**config_update.dict(exclude_none=True))
        
        # Reset models if necessary
        if "model" in changed_fields:
            model_manager.reset_llm_model()
        
        if "embedding_model" in changed_fields:
            model_manager.reset_embeddings_model()
        
        if "temperature" in changed_fields:
            model_manager.update_temperature(config_manager.get('temperature'))
        
        logger.info(f"Configuration updated successfully, changed fields: {changed_fields}")
        
        response = {
            "message": "Configuration updated successfully",
            "changed_fields": changed_fields
        }
        
        if "embedding_model" in changed_fields:
            response["warning"] = (
                "Embedding model changed. "
                "Consider rebuilding vectors for existing documents."
            )
        
        return response
    
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )


@app.get("/models", tags=["Configuration"])
async def get_models():
    """Get available models and current configuration."""
    logger.debug("Get models request")
    
    try:
        llm_models, embedding_models = get_available_models()
        
        # Fallback to defaults
        if not llm_models:
            llm_models = ["phi3", "llama3", "mistral"]
        if not embedding_models:
            embedding_models = ["nomic-embed-text"]
        
        return {
            "llm_models": llm_models,
            "embedding_models": embedding_models,
            "current_llm": config_manager.get('model'),
            "current_embedding": config_manager.get('embedding_model')
        }
    
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {
            "llm_models": ["phi3", "llama3", "mistral"],
            "embedding_models": ["nomic-embed-text"],
            "current_llm": config_manager.get('model'),
            "current_embedding": config_manager.get('embedding_model')
        }


@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get system statistics."""
    logger.debug("Statistics request")
    
    try:
        stats = vector_store.get_stats()
        
        total_size = sum(
            meta.get('size', 0) 
            for meta in metadata_manager.metadata.values()
        )
        
        doc_count = len(metadata_manager.metadata)
        avg_chunks = stats.get('total_chunks', 0) / max(1, doc_count)
        
        return {
            "total_documents": doc_count,
            "total_chunks": stats.get('total_chunks', 0),
            "total_queries": config_manager.get('total_queries'),
            "total_storage_size": total_size,
            "average_chunks_per_document": round(avg_chunks, 2),
            "last_update": stats.get('last_update'),
            "vector_store_size_mb": stats.get('vector_store_size_mb', 0)
        }
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "total_storage_size": 0,
            "average_chunks_per_document": 0.0,
            "last_update": None,
            "vector_store_size_mb": 0.0
        }


@app.post("/rebuild-vectors", tags=["Debug"])
async def rebuild_vectors():
    """Rebuild all vectors with current embedding model."""
    logger.info("Rebuild vectors request")
    
    try:
        vector_store.clear()
        results = {}
        
        for filename in list(metadata_manager.metadata.keys()):
            try:
                logger.debug(f"Rebuilding vectors for {filename}")
                
                file_path = UPLOAD_DIR / filename
                if not file_path.exists():
                    results[filename] = {"success": False, "error": "File not found"}
                    continue
                
                # Process document
                document_content = document_processor.load_document(file_path)
                documents = document_processor.split_text(document_content, filename)
                
                # Generate embeddings
                embeddings_model = model_manager.get_embeddings_model()
                texts = [doc["page_content"] for doc in documents]
                embeddings = embeddings_model.embed_documents(texts)
                
                # Add to vector store
                vector_store.add_documents(documents, embeddings)
                
                # Update metadata
                metadata = metadata_manager.get(filename) or {}
                metadata["chunks"] = len(documents)
                metadata_manager.add(filename, metadata)
                
                results[filename] = {"success": True, "chunks": len(documents)}
                logger.debug(f"Successfully rebuilt {filename}: {len(documents)} chunks")
            
            except Exception as e:
                logger.error(f"Error rebuilding {filename}: {e}")
                results[filename] = {"success": False, "error": str(e)}
        
        vector_store.save()
        
        success_count = sum(1 for result in results.values() if result["success"])
        
        logger.info(
            f"Rebuild completed: {success_count}/{len(results)} documents successful"
        )
        
        return {
            "message": (
                f"Rebuild completed: {success_count}/{len(results)} "
                f"documents processed successfully"
            ),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error rebuilding vectors: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild vectors: {str(e)}"
        )


if __name__ == "__main__":
    logger.info("Starting RAG Backend Server on port 8000...")
    uvicorn.run(
        "rag_backend_universal:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
