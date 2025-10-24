import os
import json
import logging
import traceback
import re
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)
from langchain_ollama import OllamaEmbeddings

from vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
METADATA_FILE = VECTOR_DIR / "metadata.json"
CONFIG_FILE = VECTOR_DIR / "config.json"
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = None
    top_k: int = Field(4, ge=1, le=20)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    stream: bool = False


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

# ============================================================================
# OLLAMA LLM
# ============================================================================

class OllamaLLM:
    """Universal Ollama LLM client with automatic endpoint detection"""
    
    CHAT_MODEL_PATTERNS = [
        'llama3', 'llama-3', 'gemma', 'qwen', 'mistral', 'mixtral',
        'phi3', 'phi-3', 'command', 'deepseek', 'llava', 'openchat',
        'solar', 'yi', 'nous', 'dolphin', 'orca', 'vicuna', 'wizardlm'
    ]
    
    def __init__(
        self, 
        model: str, 
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        timeout: int = 120,
        cold_start_timeout: int = 600
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.cold_start_timeout = cold_start_timeout
        self.base_url = base_url.rstrip('/')
        self.endpoint_type = None
        self.model_loaded = False
        self.model_info = None
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Fetch model information from Ollama's show API"""
        if self.model_info is not None:
            return self.model_info
        
        try:
            show_url = f"{self.base_url}/api/show"
            payload = {"name": self.model}
            response = requests.post(show_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.model_info = response.json()
                return self.model_info
            return {}
        except Exception:
            return {}
    
    def _detect_endpoint_from_model_info(self) -> Optional[str]:
        """Detect endpoint type from model information"""
        model_info = self._get_model_info()
        
        if not model_info:
            return None
        
        template = model_info.get('template', '').lower()
        modelfile = model_info.get('modelfile', '').lower()
        
        chat_indicators = [
            '{{.system}}', '{{.prompt}}', '<|im_start|>', '<|start_header_id|>',
            '[inst]', '<|user|>', '<|assistant|>', 'chatml', 'chat_template'
        ]
        
        if any(indicator in template for indicator in chat_indicators):
            return "chat"
        
        if 'chat' in modelfile or any(indicator in modelfile for indicator in chat_indicators):
            return "chat"
        
        parameters = model_info.get('parameters', '')
        if 'chat' in parameters.lower():
            return "chat"
        
        return None
    
    def _detect_endpoint_from_name(self) -> str:
        """Fallback: detect endpoint from model name patterns"""
        model_lower = self.model.lower()
        
        for pattern in self.CHAT_MODEL_PATTERNS:
            if pattern in model_lower:
                return "chat"
        
        return "generate"
    
    def _detect_endpoint(self) -> None:
        """Detect which endpoint the model supports"""
        if self.endpoint_type is not None:
            return
        
        detected_type = self._detect_endpoint_from_model_info()
        
        if detected_type is None:
            detected_type = self._detect_endpoint_from_name()
        
        self.endpoint_type = detected_type
    
    def _verify_endpoint_with_minimal_call(self) -> None:
        """Verify endpoint works with a minimal test call"""
        if self.model_loaded:
            return
        
        try:
            url, payload = self._build_payload("test", stream=False)
            
            if self.endpoint_type == "chat":
                payload["messages"] = [{"role": "user", "content": "Hi"}]
                payload["options"] = {"num_predict": 1}
            else:
                payload["prompt"] = "Hi"
                payload["options"] = {"num_predict": 1}
            
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                self.model_loaded = True
            elif response.status_code == 404 and self.endpoint_type == "chat":
                self.endpoint_type = "generate"
                self.model_loaded = False
        except Exception:
            pass
    
    def _build_payload(self, prompt: str, stream: bool) -> tuple[str, dict]:
        """Build request payload based on endpoint type"""
        self._detect_endpoint()
        
        common_options = {"temperature": self.temperature}
        
        if self.endpoint_type == "chat":
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "options": common_options
            }
        else:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": common_options
            }
        
        return url, payload
    
    def _extract_content(self, data: dict) -> str:
        """Extract content from response based on endpoint type"""
        if self.endpoint_type == "chat":
            return data.get("message", {}).get("content", "")
        return data.get("response", "")

    def _get_timeout(self, is_first_call: bool = False) -> int:
        """Get the timeout value"""
        if is_first_call or not self.model_loaded:
            return self.cold_start_timeout
        return self.timeout

    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt"""
        try:
            if not self.model_loaded:
                self._verify_endpoint_with_minimal_call()
            
            url, payload = self._build_payload(prompt, stream=False)
            current_timeout = self._get_timeout()

            response = requests.post(url, json=payload, timeout=current_timeout)
            
            if response.status_code == 404 and self.endpoint_type == "chat":
                self.endpoint_type = "generate"
                url, payload = self._build_payload(prompt, stream=False)
                response = requests.post(url, json=payload, timeout=current_timeout)
            
            response.raise_for_status()
            self.model_loaded = True
            return self._extract_content(response.json())
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Model error: {str(e)}")
        except requests.exceptions.Timeout:
            raise ValueError(f"Request timed out after {current_timeout}s")
        except Exception as e:
            raise ValueError(f"Failed to communicate with model: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response with proper connection cleanup"""
        response = None
        try:
            if not self.model_loaded:
                self._verify_endpoint_with_minimal_call()
            
            url, payload = self._build_payload(prompt, stream=True)
            current_timeout = self._get_timeout()
            
            response = requests.post(url, json=payload, timeout=current_timeout, stream=True)
            
            if response.status_code == 404 and self.endpoint_type == "chat":
                response.close()
                self.endpoint_type = "generate"
                url, payload = self._build_payload(prompt, stream=True)
                response = requests.post(url, json=payload, timeout=current_timeout, stream=True)
            
            response.raise_for_status()
            first_chunk = True
            
            try:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = self._extract_content(data)
                            
                            if first_chunk and content:
                                self.model_loaded = True
                                first_chunk = False

                            if content:
                                yield content
                            
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            finally:
                response.close()
                
        except requests.exceptions.Timeout:
            error_msg = f"Streaming request timed out after {current_timeout}s. Please try again."
            logger.error(error_msg)
            raise ValueError(error_msg)
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Streaming error: {str(e)}")
        except (BrokenPipeError, ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            logger.warning(f"Stream interrupted: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to stream: {str(e)}")
        finally:
            if response is not None:
                response.close()

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """Centralized configuration management"""
    
    DEFAULT_CONFIG = {
        'model': 'phi3',
        'embedding_model': 'nomic-embed-text',
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'temperature': 0.7,
        'timeout': 120,
        'cold_start_timeout': 600,
        'total_queries': 0
    }
    
    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self) -> None:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
            else:
                self.save()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save(self) -> None:
        """Save configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update(self, **kwargs) -> List[str]:
        """Update configuration and return list of changed fields"""
        changed = []
        for key, value in kwargs.items():
            if value is not None and key in self.config and self.config[key] != value:
                self.config[key] = value
                changed.append(key)
        
        if changed:
            self.save()
        
        return changed
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def increment_queries(self) -> None:
        """Increment query counter"""
        self.config['total_queries'] += 1
        self.save()

# ============================================================================
# METADATA MANAGER
# ============================================================================

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

# ============================================================================
# MODEL MANAGER
# ============================================================================

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
                
                # Test the model with a simple embedding
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

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Handle document loading and processing"""
    
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader
    }
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def load_document(self, file_path: Path) -> List[str]:
        """Load document content based on file type"""
        suffix = file_path.suffix.lower()
        
        if suffix not in self.LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        loader = self.LOADERS[suffix](str(file_path))
        documents = loader.load()
        content = [doc.page_content for doc in documents]
        
        return content
    
    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        """Split text content into chunks"""
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_llm_response(text: str) -> str:
    """Clean LLM response by removing reasoning tags"""
    patterns = [
        r'<think>.*?</think>',
        r'<reasoning>.*?</reasoning>',
        r'<thought>.*?</thought>',
        r'</?(?:think|reasoning|thought)>'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_available_models() -> tuple[List[str], List[str]]:
    """Get available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            all_models = [model['name'] for model in data.get('models', [])]
            
            embedding_models = [m for m in all_models if 'embed' in m.lower() or 'nomic' in m.lower()]
            llm_models = [m for m in all_models if m not in embedding_models]
            
            return llm_models, embedding_models
    except Exception:
        pass
    
    return ["phi3", "llama3", "mistral", "deepseek-r1"], ["nomic-embed-text"]


def check_ollama_health() -> tuple[bool, str]:
    """Check if Ollama is available"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200, "Ollama available" if response.status_code == 200 else "Ollama not responding"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def validate_file(filename: str, file_size: int) -> None:
    """Validate uploaded file"""
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size / (1024*1024):.1f} MB) exceeds max {MAX_FILE_SIZE_MB} MB"
        )

# ============================================================================
# INITIALIZE GLOBAL OBJECTS
# ============================================================================

config_manager = ConfigManager()
metadata_manager = MetadataManager()
vector_store = VectorStore()
model_manager = ModelManager(config_manager)
document_processor = DocumentProcessor(config_manager)

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting RAG Application...")
    
    # Pre-initialize embeddings model to avoid first upload failure
    try:
        model_manager.get_embeddings_model()
        logger.info("Embeddings model pre-initialized successfully")
    except Exception as e:
        logger.warning(f"Could not pre-initialize embeddings model: {e}")
    
    try:
        vector_store.load()
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load vector store: {e}")
    
    logger.info("RAG Application started successfully")
    
    yield
    
    logger.info("Shutting down RAG Application...")


app = FastAPI(
    title="RAG Assistant API",
    description="Production-ready RAG system with Ollama",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Check system health and configuration"""
    try:
        ollama_available, ollama_message = check_ollama_health()
        stats = vector_store.get_stats()
        
        return {
            "status": "healthy",
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
        return {
            "status": "unhealthy",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    logger.info(f"Upload request: {file.filename}")
    
    content = await file.read()
    file_size = len(content)
    
    validate_file(file.filename, file_size)
    
    if metadata_manager.exists(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists"
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
        
        document_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(document_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        embeddings_model = model_manager.get_embeddings_model()
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        vector_store.add_documents(documents, embeddings)
        
        metadata_manager.add(file.filename, {
            "filename": file.filename,
            "size": file_size,
            "chunks": len(documents),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        })
        
        vector_store.save()
        
        logger.info(f"Processed {file.filename}: {len(documents)} chunks")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=file_size,
            message=f"Document processed successfully into {len(documents)} chunks"
        )
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        if file_path.exists():
            file_path.unlink()
        metadata_manager.remove(file.filename)
        
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query", tags=["Query"])
async def query_documents(request: Request, query: QueryRequest):
    """Query the document collection with streaming support"""
    logger.info(f"Query: '{query.question[:50]}...'")
    start_time = datetime.now()
    
    try:
        if vector_store.get_stats().get("total_chunks", 0) == 0:
            raise HTTPException(status_code=400, detail="No documents available. Upload documents first.")
        
        embeddings_model = model_manager.get_embeddings_model()
        query_embedding = embeddings_model.embed_query(query.question)
        similar_docs = vector_store.similarity_search(query_embedding, k=query.top_k)
        
        if not similar_docs:
            raise HTTPException(status_code=400, detail="No relevant documents found")
        
        context_parts = []
        sources = []
        similarity_scores = []
        
        for doc, score in similar_docs:
            context_parts.append(f"Source: {doc['metadata']['source']}\nContent: {doc['page_content']}")
            sources.append(doc['metadata']['source'])
            similarity_scores.append(float(score))
        
        context = "\n\n".join(context_parts)
        
        llm = model_manager.get_llm_model(query.model, query.temperature)
        
        unique_sources = list(set(sources))
        doc_identity = unique_sources[0] if len(unique_sources) == 1 else "your documents"
        prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person.

Your content includes:
{context}

The user asks: {query.question}

Respond naturally as the document. Your response:"""
        
        if query.stream:
            async def generate():
                stream_generator = None
                try:
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": similarity_scores,
                        "model_used": llm.model,
                        "endpoint_type": llm.endpoint_type,
                        "type": "metadata"
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"

                    stream_generator = llm.stream(prompt)
                    for chunk in stream_generator:
                        if await request.is_disconnected():
                            logger.info("Client disconnected, stopping stream")
                            break
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                    processing_time = (datetime.now() - start_time).total_seconds()
                    completion = {"type": "done", "processing_time": processing_time}
                    yield f"data: {json.dumps(completion)}\n\n"
                    
                    config_manager.increment_queries()
                    logger.info(f"Query completed in {processing_time:.2f}s")
                except (BrokenPipeError, ConnectionError, ConnectionResetError) as e:
                    logger.warning(f"Client connection lost during streaming: {e}")
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    try:
                        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                    except (BrokenPipeError, ConnectionError, ConnectionResetError):
                        logger.warning("Cannot send error to client - connection already closed")
                finally:
                    if stream_generator is not None:
                        try:
                            stream_generator.close()
                        except Exception:
                            pass
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            answer = llm.invoke(prompt)
            answer = clean_llm_response(answer)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            config_manager.increment_queries()
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(similar_docs),
                "similarity_scores": similarity_scores,
                "processing_time": processing_time,
                "model_used": llm.model
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all uploaded documents"""
    return [DocumentInfo(**meta) for meta in metadata_manager.list_all()]


@app.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a specific document"""
    logger.info(f"Delete request: {filename}")
    
    if not metadata_manager.exists(filename):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    
    try:
        vector_store.remove_documents_by_source(filename)
        metadata_manager.remove(filename)
        
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        vector_store.save()
        
        logger.info(f"Deleted: {filename}")
        return {"message": f"Document '{filename}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.delete("/clear", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents and embeddings"""
    try:
        vector_store.clear()
        metadata_manager.clear()
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        vector_store.save()
        
        logger.info("Cleared all documents")
        return {"message": "All documents cleared successfully", "cleared": True}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@app.post("/configure", tags=["Configuration"])
async def configure_system(config_update: ModelConfig):
    """Update system configuration"""
    try:
        changed_fields = config_manager.update(**config_update.dict(exclude_none=True))
        
        if "model" in changed_fields:
            model_manager.reset_llm_cache()
        
        if "embedding_model" in changed_fields:
            model_manager.reset_embeddings_model()
        
        logger.info(f"Configuration updated: {changed_fields}")
        
        response = {
            "message": "Configuration updated successfully",
            "changed_fields": changed_fields
        }
        
        if "embedding_model" in changed_fields:
            response["warning"] = "Embedding model changed. Consider rebuilding vectors."
        
        return response
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@app.get("/models", tags=["Configuration"])
async def get_models():
    """Get available models"""
    try:
        llm_models, embedding_models = get_available_models()
        
        return {
            "ollama": {
                "llm_models": llm_models,
                "embedding_models": embedding_models,
            },
            "current_config": {
                "model": config_manager.get('model'),
                "embedding_model": config_manager.get('embedding_model')
            }
        }
    except Exception as e:
        return {
            "ollama": {
                "llm_models": ["phi3", "llama3", "mistral", "deepseek-r1"],
                "embedding_models": ["nomic-embed-text"],
            },
            "current_config": {
                "model": config_manager.get('model'),
                "embedding_model": config_manager.get('embedding_model')
            }
        }


@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get system statistics"""
    try:
        stats = vector_store.get_stats()
        doc_count = len(metadata_manager.metadata)
        total_size = sum(meta.get('size', 0) for meta in metadata_manager.metadata.values())
        avg_chunks = stats.get('total_chunks', 0) / max(1, doc_count)
        
        return {
            "total_documents": doc_count,
            "total_chunks": stats.get('total_chunks', 0),
            "total_queries": config_manager.get('total_queries'),
            "total_storage_size": total_size,
            "average_chunks_per_document": round(avg_chunks, 2),
            "last_update": stats.get('last_update'),
            "vector_store_size_mb": round(stats.get('vector_store_size', 0) / (1024 * 1024), 2)
        }
    except Exception:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "total_storage_size": 0,
            "average_chunks_per_document": 0.0
        }


@app.post("/rebuild-vectors", tags=["Maintenance"])
async def rebuild_vectors():
    """Rebuild all vectors with current embedding model"""
    logger.info("Rebuild vectors request")
    
    try:
        vector_store.clear()
        results = {}
        embeddings_model = model_manager.get_embeddings_model()
        
        for filename in list(metadata_manager.metadata.keys()):
            try:
                file_path = UPLOAD_DIR / filename
                if not file_path.exists():
                    results[filename] = {"success": False, "error": "File not found"}
                    continue
                
                document_content = document_processor.load_document(file_path)
                documents = document_processor.split_text(document_content, filename)
                
                texts = [doc["page_content"] for doc in documents]
                embeddings = embeddings_model.embed_documents(texts)
                
                vector_store.add_documents(documents, embeddings)
                
                metadata = metadata_manager.get(filename) or {}
                metadata["chunks"] = len(documents)
                metadata_manager.add(filename, metadata)
                
                results[filename] = {"success": True, "chunks": len(documents)}
            except Exception as e:
                logger.error(f"Error rebuilding {filename}: {e}")
                results[filename] = {"success": False, "error": str(e)}
        
        vector_store.save()
        
        success_count = sum(1 for result in results.values() if result["success"])
        
        logger.info(f"Rebuild completed: {success_count}/{len(results)} successful")
        
        return {
            "message": f"Rebuild completed: {success_count}/{len(results)} documents processed",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error rebuilding vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild vectors: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting RAG Backend Server on port 8000...")
    uvicorn.run(
        "rag_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
