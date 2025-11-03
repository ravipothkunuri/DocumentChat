"""
DocumentChat Backend - AI-Powered Document Processing and Query System
FastAPI application providing semantic search and conversational AI over documents
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, AsyncIterator
from contextlib import asynccontextmanager
from functools import wraps

import numpy as np
import httpx
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
FIXED_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for document queries with streaming support."""
    question: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = None
    top_k: int = Field(4, ge=1, le=20)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False

class DocumentUploadResponse(BaseModel):
    """Response model for successful document uploads."""
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str

class DocumentInfo(BaseModel):
    """Metadata model for stored documents."""
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str

# ============================================================================
# UTILITIES & DECORATORS
# ============================================================================

def handle_errors(operation_name: str):
    """
    Decorator for consistent error handling across operations.
    
    Args:
        operation_name: Human-readable name of the operation for logging
        
    Returns:
        Decorated function with automatic error logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}")
                raise
        return wrapper
    return decorator

def http_response(status_code: int, message: str, **extra_data) -> Dict[str, Any]:
    """
    Create standardized HTTP response dictionary.
    
    Args:
        status_code: HTTP status code
        message: Response message
        **extra_data: Additional key-value pairs for response
        
    Returns:
        Dictionary with status, message, and extra data
    """
    return {"status": "success" if status_code < 400 else "error", 
            "message": message, **extra_data}

def validate_file(filename: str, file_size: int) -> None:
    """
    Validate uploaded file against extension and size constraints.
    
    Args:
        filename: Name of the uploaded file
        file_size: Size of file in bytes
        
    Raises:
        HTTPException: If file validation fails
    """
    file_path = Path(filename)
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {file_path.suffix}")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(400, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

def check_ollama_health() -> Tuple[bool, str]:
    """
    Check if Ollama service is available and responding.
    
    Returns:
        Tuple of (is_available, status_message)
    """
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200, "Available" if response.status_code == 200 else f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# ============================================================================
# GENERIC JSON STORE
# ============================================================================

class JSONStore:
    """
    Generic persistent JSON storage with automatic error handling.
    Consolidates file-based configuration and metadata management.
    """
    
    def __init__(self, filepath: Path, defaults: Optional[Dict] = None):
        """
        Initialize JSON store with file path and optional defaults.
        
        Args:
            filepath: Path to JSON storage file
            defaults: Default values if file doesn't exist
        """
        self.filepath = filepath
        self.data = defaults.copy() if defaults else {}
        self.load()
    
    @handle_errors("JSON load")
    def load(self) -> None:
        """Load data from JSON file. Creates file with defaults if missing."""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                loaded_data = json.load(f)
                self.data.update(loaded_data)
    
    @handle_errors("JSON save")
    def save(self) -> None:
        """Persist current data to JSON file."""
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get(self, key: str, default=None) -> Any:
        """Get value by key with optional default."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value and persist to disk."""
        self.data[key] = value
        self.save()
    
    def increment(self, key: str, amount: int = 1) -> None:
        """Atomically increment numeric value."""
        self.data[key] = self.data.get(key, 0) + amount
        self.save()
    
    def remove(self, key: str) -> bool:
        """Remove key if exists and persist."""
        if key in self.data:
            del self.data[key]
            self.save()
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in store."""
        return key in self.data
    
    def all_values(self) -> List[Any]:
        """Get all values as list."""
        return list(self.data.values())

# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore:
    """
    In-memory vector store with cosine similarity search and JSON persistence.
    Manages document embeddings and enables semantic retrieval.
    """
    
    def __init__(self, storage_path: Path = VECTOR_DIR / "vectors.json"):
        """
        Initialize vector store with optional persistence path.
        
        Args:
            storage_path: Path for JSON persistence of vectors and documents
        """
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dimensions: Optional[int] = None
        self.last_update: Optional[str] = None
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of document dictionaries with content and metadata
            embeddings: List of embedding vectors (must match document count)
            
        Raises:
            ValueError: If documents/embeddings are invalid or dimensions mismatch
        """
        if not documents or not embeddings or len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must be non-empty and equal length")
        
        embedding_array = np.array(embeddings, dtype=np.float32)
        
        if self.embedding_dimensions is None:
            self.embedding_dimensions = embedding_array.shape[1]
        elif embedding_array.shape[1] != self.embedding_dimensions:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimensions}, got {embedding_array.shape[1]}")
        
        self.documents.extend(documents)
        self.embeddings = (embedding_array if self.embeddings is None 
                          else np.vstack([self.embeddings, embedding_array]))
        self.last_update = datetime.now().isoformat()
    
    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find top-k most similar documents using cosine similarity.
        
        Args:
            query_embedding: Query vector to search with
            k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples sorted by relevance
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Normalize query vector (avoid division by zero)
        query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)
        query_normalized = query_array / np.where(query_norm == 0, 1, query_norm)
        
        # Normalize stored embeddings
        embeddings_norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        embeddings_normalized = self.embeddings / np.where(embeddings_norm == 0, 1, embeddings_norm)
        
        # Compute cosine similarities
        similarities = np.dot(query_normalized, embeddings_normalized.T)[0]
        
        # Get top-k indices
        top_k = min(k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.documents[idx], float(similarities[idx])) for idx in top_indices]
    
    def remove_documents_by_source(self, source: str) -> None:
        """
        Remove all documents from a specific source file.
        
        Args:
            source: Source filename to remove documents from
        """
        indices = [i for i, doc in enumerate(self.documents) 
                  if doc.get('metadata', {}).get('source') == source]
        
        if not indices:
            return
        
        for i in reversed(indices):
            del self.documents[i]
        
        if self.embeddings is not None:
            mask = np.ones(self.embeddings.shape[0], dtype=bool)
            mask[indices] = False
            self.embeddings = self.embeddings[mask] if mask.any() else None
            if self.embeddings is None:
                self.embedding_dimensions = None
        
        self.last_update = datetime.now().isoformat()
    
    def clear(self) -> None:
        """Remove all documents and reset store to empty state."""
        self.documents = []
        self.embeddings = None
        self.embedding_dimensions = None
        self.last_update = datetime.now().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current store statistics."""
        return {
            "total_chunks": len(self.documents),
            "last_update": self.last_update
        }
    
    @handle_errors("Vector store save")
    def save(self) -> None:
        """Persist vectors and documents to JSON file."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "embedding_dimensions": self.embedding_dimensions,
            "last_update": self.last_update
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @handle_errors("Vector store load")
    def load(self) -> None:
        """Load vectors and documents from JSON file."""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        self.documents = data.get("documents", [])
        self.embedding_dimensions = data.get("embedding_dimensions")
        self.last_update = data.get("last_update")
        
        if embeddings_data := data.get("embeddings"):
            self.embeddings = np.array(embeddings_data, dtype=np.float32)
        
        logger.info(f"Loaded {len(self.documents)} documents from vector store")

# ============================================================================
# OLLAMA LLM CLIENT
# ============================================================================

class AsyncOllamaLLM:
    """
    Async HTTP client for Ollama LLM with streaming support.
    Handles chat completions and manages connection lifecycle.
    """
    
    def __init__(self, model: str = FIXED_MODEL, base_url: str = OLLAMA_BASE_URL, temperature: float = 0.7):
        """
        Initialize Ollama client with model configuration.
        
        Args:
            model: Name of Ollama model to use
            base_url: Base URL of Ollama API service
            temperature: Sampling temperature (0.0-2.0)
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream LLM response tokens as they are generated.
        
        Args:
            prompt: Input prompt for the model
            
        Yields:
            String chunks of the generated response
            
        Raises:
            ValueError: If request fails or times out
        """
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"temperature": self.temperature}
            }
            
            async with self.client.stream('POST', url, json=payload, timeout=180.0) as response:
                if response.status_code != 200:
                    raise ValueError(f"Ollama returned HTTP {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if content := data.get("message", {}).get("content", ""):
                                yield content
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ReadTimeout:
            raise ValueError("Request timed out - model may be unresponsive")
        except httpx.ConnectError:
            raise ValueError("Cannot connect to Ollama service")
        except Exception as e:
            raise ValueError(f"Stream error: {str(e)}")
    
    async def close(self) -> None:
        """Close underlying HTTP client connection."""
        await self.client.aclose()

# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Manages AI model instances with lazy loading.
    Handles both embedding and LLM models.
    """
    
    def __init__(self, config: JSONStore):
        """
        Initialize model manager with configuration store.
        
        Args:
            config: JSONStore containing model configuration
        """
        self.config = config
        self._embeddings_model: Optional[OllamaEmbeddings] = None
        self._llm: Optional[AsyncOllamaLLM] = None
    
    @property
    def embeddings_model(self) -> OllamaEmbeddings:
        """Get or create embeddings model instance (lazy loaded)."""
        if self._embeddings_model is None:
            model_name = self.config.get('embedding_model')
            self._embeddings_model = OllamaEmbeddings(
                model=model_name, 
                base_url=OLLAMA_BASE_URL
            )
        return self._embeddings_model
    
    @property
    def llm(self) -> AsyncOllamaLLM:
        """Get or create LLM instance (lazy loaded)."""
        if self._llm is None:
            self._llm = AsyncOllamaLLM(
                model=FIXED_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=self.config.get('temperature', 0.7)
            )
        return self._llm
    
    async def cleanup(self) -> None:
        """Clean up model resources and close connections."""
        if self._llm:
            await self._llm.close()

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """
    Handles document loading and text chunking for RAG pipeline.
    Supports PDF, TXT, and DOCX formats.
    """
    
    LOADERS = {
        '.pdf': PyMuPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader
    }
    
    def __init__(self, config: JSONStore):
        """
        Initialize processor with configuration.
        
        Args:
            config: JSONStore with chunk_size and chunk_overlap settings
        """
        self.config = config
    
    def load_document(self, file_path: Path) -> List[str]:
        """
        Load document content from file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of page contents as strings
            
        Raises:
            ValueError: If file type is unsupported
        """
        suffix = file_path.suffix.lower()
        if suffix not in self.LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        loader = self.LOADERS[suffix](str(file_path))
        documents = loader.load()
        return [doc.page_content for doc in documents]
    
    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        """
        Split document content into chunks for embedding.
        
        Args:
            content: List of document page contents
            filename: Source filename for metadata
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        full_text = "\n\n".join(content)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size', 1000),
            chunk_overlap=self.config.get('chunk_overlap', 200),
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(full_text)
        return [
            {
                "page_content": chunk,
                "metadata": {"source": filename, "chunk_id": i}
            }
            for i, chunk in enumerate(chunks)
        ]

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize stores and managers
config_store = JSONStore(
    VECTOR_DIR / "config.json",
    defaults={
        'model': FIXED_MODEL,
        'embedding_model': DEFAULT_EMBEDDING_MODEL,
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'temperature': 0.7,
        'total_queries': 0
    }
)

metadata_store = JSONStore(VECTOR_DIR / "metadata.json")
vector_store = VectorStore()
model_manager = ModelManager(config_store)
document_processor = DocumentProcessor(config_store)

# Load persisted vector store
try:
    vector_store.load()
    logger.info("Vector store loaded successfully")
except Exception as e:
    logger.warning(f"Could not load vector store: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown hooks."""
    logger.info("🚀 Starting DocumentChat Application")
    yield
    logger.info("📻 Shutting down gracefully")
    await model_manager.cleanup()

app = FastAPI(
    title="DocumentChat Backend API",
    version="1.0.0",
    description="AI-powered conversational interface for document understanding and question answering",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Check system health and return current configuration.
    
    Returns comprehensive status including Ollama availability,
    document count, and processing statistics.
    """
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
            "model": FIXED_MODEL,
            "embedding_model": config_store.get('embedding_model'),
            "chunk_size": config_store.get('chunk_size')
        },
        "document_count": len(metadata_store.data),
        "total_chunks": stats.get("total_chunks", 0),
        "total_queries": config_store.get('total_queries')
    }

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process document for conversational querying.
    
    Validates file, extracts text, generates embeddings, and stores
    in vector database. Prevents duplicate uploads.
    
    Args:
        file: Uploaded file (PDF, TXT, or DOCX)
        
    Returns:
        Upload status with chunk count and processing details
        
    Raises:
        HTTPException: If validation fails or processing errors occur
    """
    content = await file.read()
    validate_file(file.filename, len(content))
    
    if metadata_store.exists(file.filename):
        raise HTTPException(400, f"Document '{file.filename}' already exists")
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save uploaded file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Process document
        doc_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(doc_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        texts = [doc["page_content"] for doc in documents]
        embeddings = model_manager.embeddings_model.embed_documents(texts)
        
        # Store in vector database
        vector_store.add_documents(documents, embeddings)
        
        # Save metadata
        metadata_store.set(file.filename, {
            "filename": file.filename,
            "size": len(content),
            "chunks": len(documents),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        })
        
        vector_store.save()
        logger.info(f"Successfully processed {file.filename}: {len(documents)} chunks")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=len(content),
            message=f"Processed into {len(documents)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Processing failed for {file.filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        metadata_store.remove(file.filename)
        raise HTTPException(500, f"Processing failed: {str(e)}")

@router.post("/query")
async def query_documents(request: Request, query: QueryRequest):
    """
    Query documents using semantic search and LLM generation.
    
    Finds relevant document chunks via similarity search, constructs
    context, and generates response using Ollama LLM. Supports streaming.
    
    Args:
        request: FastAPI request object for disconnect detection
        query: Query parameters including question and settings
        
    Returns:
        StreamingResponse (if stream=True) or JSON with answer and metadata
        
    Raises:
        HTTPException: If no documents available or query processing fails
    """
    start_time = datetime.now()
    
    try:
        if vector_store.get_stats().get("total_chunks", 0) == 0:
            raise HTTPException(400, "No documents available for querying")
        
        # Semantic search
        query_embedding = model_manager.embeddings_model.embed_query(query.question)
        similar_docs = vector_store.similarity_search(query_embedding, k=query.top_k)
        
        if not similar_docs:
            raise HTTPException(400, "No relevant documents found")
        
        # Build context
        context_parts = []
        sources = []
        scores = []
        
        for doc, score in similar_docs:
            context_parts.append(
                f"Source: {doc['metadata']['source']}\n"
                f"Content: {doc['page_content']}"
            )
            sources.append(doc['metadata']['source'])
            scores.append(float(score))
        
        context = "\n\n".join(context_parts)
        unique_sources = list(set(sources))
        doc_identity = unique_sources[0] if len(unique_sources) == 1 else "your documents"
        
        # Generate prompt
        prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person.

Your content:
{context}

User asks: {query.question}

Your response:"""
        
        llm = model_manager.llm
        
        # Stream response
        if query.stream:
            async def generate():
                try:
                    # Send metadata first
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": scores,
                        "model_used": FIXED_MODEL,
                        "type": "metadata"
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"
                    
                    # Stream content
                    async for chunk in llm.astream(prompt):
                        if await request.is_disconnected():
                            break
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                    
                    # Send completion
                    processing_time = (datetime.now() - start_time).total_seconds()
                    yield f"data: {json.dumps({'type': 'done', 'processing_time': processing_time})}\n\n"
                    
                    config_store.increment('total_queries')
                    
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Non-streaming response
        else:
            full_response = ""
            async for chunk in llm.astream(prompt):
                full_response += chunk
            
            processing_time = (datetime.now() - start_time).total_seconds()
            config_store.increment('total_queries')
            
            return {
                "answer": full_response,
                "sources": sources,
                "chunks_used": len(similar_docs),
                "similarity_scores": scores,
                "processing_time": processing_time,
                "model_used": FIXED_MODEL
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all uploaded documents with metadata.
    
    Returns:
        List of document information including size, chunks, and status
    """
    return [DocumentInfo(**meta) for meta in metadata_store.all_values()]

@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete document and remove from vector store.
    
    Args:
        filename: Name of document to delete
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If document not found or deletion fails
    """
    if not metadata_store.exists(filename):
        raise HTTPException(404, f"Document '{filename}' not found")
    
    try:
        vector_store.remove_documents_by_source(filename)
        metadata_store.remove(filename)
        
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        vector_store.save()
        return http_response(200, f"Document '{filename}' deleted successfully")
        
    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")

@router.delete("/clear")
async def clear_all_documents():
    """
    Clear all documents from system.
    
    Removes all documents, vectors, and metadata. Cannot be undone.
    
    Returns:
        Confirmation of cleared state
        
    Raises:
        HTTPException: If clearing operation fails
    """
    try:
        vector_store.clear()
        metadata_store.data = {}
        metadata_store.save()
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        vector_store.save()
        return http_response(200, "All documents cleared", cleared=True)
        
    except Exception as e:
        raise HTTPException(500, f"Clear operation failed: {str(e)}")

@router.get("/stats")
async def get_statistics():
    """
    Get system usage statistics.
    
    Returns:
        Statistics including document count, chunks, queries, and storage
    """
    stats = vector_store.get_stats()
    doc_count = len(metadata_store.data)
    total_size = sum(meta.get('size', 0) for meta in metadata_store.data.values())
    
    return {
        "total_documents": doc_count,
        "total_chunks": stats.get('total_chunks', 0),
        "total_queries": config_store.get('total_queries'),
        "total_storage_size": total_size,
        "average_chunks_per_document": round(stats.get('total_chunks', 0) / max(1, doc_count), 2),
        "last_update": stats.get('last_update')
    }

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)