"""
DocumentChat Backend - AI Document Processing & Query System (Refactored)

This module implements a FastAPI-based backend service using the unified
DocumentStore for simplified storage management. All document operations
now go through a single facade interface.

Key Changes:
    - Uses DocumentStore instead of separate JSONStore/VectorStore instances
    - Simplified code with unified storage API
    - Same functionality with cleaner architecture

Author: Your Name
Version: 2.0.0
"""

import json
import logging
from datetime import datetime
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, AsyncIterator
from contextlib import asynccontextmanager
import httpx
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader

# Import storage classes
from storage import DocumentStore

from configuration import (
    FALLBACK_BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL,
    MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, UPLOAD_DIR, VECTOR_DIR,
    MAX_FILE_SIZE_BYTES
)

# Configuration from environment variables
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
API_BASE_URL = os.environ.get("OLLAMA_BASE_URL", FALLBACK_BASE_URL)
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_MODEL)

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models - Request/Response Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """
    Schema for document query requests.
    
    Attributes:
        question: The user's question (1-5000 characters)
        model: Optional model override (uses default if None)
        top_k: Number of similar chunks to retrieve (1-20)
        temperature: LLM sampling temperature (0.0-2.0)
        stream: Whether to stream the response
    """
    question: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = None
    top_k: int = Field(4, ge=1, le=20)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False


class DocumentUploadResponse(BaseModel):
    """Response schema for successful document uploads."""
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str


class DocumentInfo(BaseModel):
    """Schema for document metadata."""
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str


# =============================================================================
# Utility Functions
# =============================================================================

def http_response(status_code: int, message: str, **extra_data) -> Dict[str, Any]:
    """
    Create a standardized HTTP response dictionary.
    
    Args:
        status_code: HTTP status code
        message: Response message
        **extra_data: Additional key-value pairs to include
        
    Returns:
        Dictionary with status, message, and extra data
    """
    return {
        "status": "success" if status_code < 400 else "error",
        "message": message,
        **extra_data
    }


def validate_file(filename: str, file_size: int) -> None:
    """
    Validate uploaded file type and size.
    
    Args:
        filename: Name of the file
        file_size: Size of the file in bytes
        
    Raises:
        HTTPException: If file type is unsupported or size exceeds limit
    """
    file_path = Path(filename)
    
    # Check file extension
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {file_path.suffix}")
    
    # Check file size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(400, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")


def check_ollama_health() -> Tuple[bool, str]:
    """
    Check if Ollama service is available and responding.
    
    Returns:
        Tuple of (is_available, status_message)
    """
    try:
        response = httpx.get(f"{API_BASE_URL}/api/tags", timeout=5)
        is_available = response.status_code == 200
        message = "Available" if is_available else f"HTTP {response.status_code}"
        return is_available, message
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


# =============================================================================
# Async Ollama LLM Client
# =============================================================================

class AsyncOllamaLLM:
    """
    Async client for streaming responses from Ollama LLM.
    
    Provides async streaming interface to Ollama's chat API with proper
    error handling and timeout management.
    """
    
    def __init__(
        self,
        model: str = OLLAMA_CHAT_MODEL,
        base_url: str = API_BASE_URL,
        temperature: float = 0.7
    ):
        """
        Initialize the Ollama LLM client.
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: Sampling temperature
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream LLM response chunks asynchronously.
        
        Args:
            prompt: Input prompt for the LLM
            
        Yields:
            String chunks of the generated response
            
        Raises:
            ValueError: If connection fails, times out, or returns error
        """
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"temperature": self.temperature}
            }
            
            # Stream response from Ollama
            async with self.client.stream(
                'POST', url, json=payload, timeout=180.0
            ) as response:
                if response.status_code != 200:
                    raise ValueError(f"Ollama returned HTTP {response.status_code}")
                
                # Process each line of the stream
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            
                            # Extract content from message
                            if content := data.get("message", {}).get("content", ""):
                                yield content
                            
                            # Check if generation is complete
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ReadTimeout:
            raise ValueError("Request timed out")
        except httpx.ConnectError:
            raise ValueError("Cannot connect to Ollama service")
        except Exception as e:
            raise ValueError(f"Stream error: {str(e)}")

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self.client.aclose()


# =============================================================================
# Model Manager - Lazy Loading of Models
# =============================================================================

class ModelManager:
    """
    Manages lazy initialization of embeddings and LLM models.
    
    Creates models only when first accessed to optimize startup time
    and memory usage.
    """
    
    def __init__(self, document_store: DocumentStore):
        """
        Initialize the model manager.
        
        Args:
            document_store: DocumentStore instance for configuration
        """
        self.document_store = document_store
        self._embeddings_model: Optional[OllamaEmbeddings] = None
        self._llm: Optional[AsyncOllamaLLM] = None

    @property
    def embeddings_model(self) -> OllamaEmbeddings:
        """
        Get or create embeddings model.
        
        Returns:
            Initialized OllamaEmbeddings instance
        """
        if self._embeddings_model is None:
            self._embeddings_model = OllamaEmbeddings(
                model=OLLAMA_EMBED_MODEL,
                base_url=API_BASE_URL
            )
        return self._embeddings_model

    @property
    def llm(self) -> AsyncOllamaLLM:
        """
        Get or create LLM.
        
        Returns:
            Initialized AsyncOllamaLLM instance
        """
        if self._llm is None:
            self._llm = AsyncOllamaLLM(
                model=OLLAMA_CHAT_MODEL,
                base_url=API_BASE_URL,
                temperature=self.document_store.get_config('temperature', 0.7)
            )
        return self._llm

    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self._llm:
            await self._llm.close()


# =============================================================================
# Document Processor - Document Loading and Chunking
# =============================================================================

class DocumentProcessor:
    """
    Handles document loading and text splitting.
    
    Supports PDF, TXT, and DOCX formats. Uses appropriate loaders
    for each format and splits text into manageable chunks.
    """
    
    # Mapping of file extensions to LangChain loader classes
    LOADERS = {
        '.pdf': PyMuPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader
    }

    def __init__(self, document_store: DocumentStore):
        """
        Initialize the document processor.
        
        Args:
            document_store: DocumentStore instance for configuration
        """
        self.document_store = document_store

    def load_document(self, file_path: Path) -> List[str]:
        """
        Load document content from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of page contents as strings
            
        Raises:
            ValueError: If file type is not supported
        """
        suffix = file_path.suffix.lower()
        
        if suffix not in self.LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        loader = self.LOADERS[suffix](str(file_path))
        documents = loader.load()
        
        return [doc.page_content for doc in documents]

    def split_text(
        self,
        content: List[str],
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Split document content into chunks.
        
        Args:
            content: List of page contents
            filename: Source filename for metadata
            
        Returns:
            List of chunk dictionaries with page_content and metadata
        """
        # Combine all pages
        full_text = "\n\n".join(content)
        
        # Create splitter with configured parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.document_store.get_config('chunk_size', 1000),
            chunk_overlap=self.document_store.get_config('chunk_overlap', 200),
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split text into chunks
        chunks = splitter.split_text(full_text)
        
        # Create chunk dictionaries with metadata
        return [
            {
                "page_content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]


# =============================================================================
# Initialize Global Components
# =============================================================================

# Initialize unified document store
document_store = DocumentStore(VECTOR_DIR)

# Initialize model manager and document processor
model_manager = ModelManager(document_store)
document_processor = DocumentProcessor(document_store)

# Set default configuration if not already set
if document_store.get_config('model') is None:
    document_store.set_config('model', OLLAMA_CHAT_MODEL)
if document_store.get_config('embedding_model') is None:
    document_store.set_config('embedding_model', OLLAMA_EMBED_MODEL)


# =============================================================================
# FastAPI Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Handles startup and shutdown events.
    """
    logger.info("🚀 Starting DocumentChat Application")
    yield
    logger.info("📻 Shutting down gracefully")
    await model_manager.cleanup()


# Create FastAPI application
app = FastAPI(
    title="DocumentChat Backend API",
    version="2.0.0",
    description="AI-powered document understanding and question answering",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create API router
router = APIRouter()


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns system status, configuration, and statistics.
    """
    # Check Ollama service
    ollama_available, ollama_message = check_ollama_health()
    
    # Get statistics from document store
    stats = document_store.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ollama_status": {
            "available": ollama_available,
            "message": ollama_message
        },
        "configuration": {
            "model": OLLAMA_CHAT_MODEL,
            "embedding_model": document_store.get_config('embedding_model'),
            "chunk_size": document_store.get_config('chunk_size')
        },
        "document_count": stats['total_documents'],
        "total_chunks": stats['total_chunks'],
        "total_queries": stats['total_queries']
    }


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Handles file validation, content extraction, chunking,
    embedding generation, and storage.
    """
    # Read file content
    content = await file.read()
    
    # Validate file
    validate_file(file.filename, len(content))

    # Check for duplicates
    if document_store.document_exists(file.filename):
        raise HTTPException(400, f"Document '{file.filename}' already exists")

    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save uploaded file
        with open(file_path, 'wb') as f:
            f.write(content)

        # Load and process document
        doc_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(doc_content, file.filename)

        if not documents:
            raise ValueError("No content extracted from document")

        # Generate embeddings
        texts = [doc["page_content"] for doc in documents]
        embeddings = model_manager.embeddings_model.embed_documents(texts)
        
        # Store document (both metadata and vectors)
        document_store.add_document(
            filename=file.filename,
            chunks=documents,
            embeddings=embeddings,
            file_size=len(content),
            file_type=file_path.suffix[1:].lower()
        )
        
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
        
        # Cleanup on failure
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(500, f"Processing failed: {str(e)}")


@router.post("/query")
async def query_documents(request: Request, query: QueryRequest):
    """
    Query documents using natural language.
    
    Performs semantic search and generates AI responses based on
    retrieved document chunks.
    """
    start_time = datetime.now()
    
    try:
        # Get statistics to check if documents exist
        stats = document_store.get_stats()
        if stats['total_chunks'] == 0:
            raise HTTPException(400, "No documents available for querying")

        # Generate query embedding
        query_embedding = model_manager.embeddings_model.embed_query(
            query.question
        )
        
        # Search for similar documents
        similar_docs = document_store.search(query_embedding, k=query.top_k)

        if not similar_docs:
            raise HTTPException(400, "No relevant documents found")

        # Build context from retrieved chunks
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
        
        # Determine document identity for prompt
        unique_sources = list(set(sources))
        doc_identity = (
            unique_sources[0] if len(unique_sources) == 1
            else "your documents"
        )

        # Construct prompt
        prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person.

Your content:
{context}

User asks: {query.question}

Your response:"""

        llm = model_manager.llm

        # Handle streaming response
        if query.stream:
            async def generate():
                """Generator for streaming response."""
                try:
                    # Send metadata
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": scores,
                        "model_used": OLLAMA_CHAT_MODEL,
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
                    
                    # Increment query counter
                    document_store.increment_query_count()
                    
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
            
        # Handle non-streaming response
        else:
            full_response = ""
            async for chunk in llm.astream(prompt):
                full_response += chunk

            processing_time = (datetime.now() - start_time).total_seconds()
            document_store.increment_query_count()

            return {
                "answer": full_response,
                "sources": sources,
                "chunks_used": len(similar_docs),
                "similarity_scores": scores,
                "processing_time": processing_time,
                "model_used": OLLAMA_CHAT_MODEL
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all uploaded documents.
    
    Returns list of document metadata.
    """
    return [DocumentInfo(**meta) for meta in document_store.list_documents()]


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a specific document.
    
    Removes document chunks, metadata, and the uploaded file.
    """
    if not document_store.document_exists(filename):
        raise HTTPException(404, f"Document '{filename}' not found")
    
    try:
        # Remove from document store (handles both metadata and vectors)
        document_store.remove_document(filename)
        
        # Remove file from disk
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        return http_response(200, f"Document '{filename}' deleted successfully")
        
    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")


@router.delete("/clear")
async def clear_all_documents():
    """
    Clear all documents from the system.
    
    Removes all document data and uploaded files.
    """
    try:
        # Clear document store (handles both metadata and vectors)
        document_store.clear_all()
        
        # Remove all uploaded files
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        return http_response(200, "All documents cleared", cleared=True)
        
    except Exception as e:
        raise HTTPException(500, f"Clear operation failed: {str(e)}")


@router.get("/stats")
async def get_statistics():
    """
    Get system-wide statistics.
    
    Returns aggregated statistics from document store.
    """
    return document_store.get_stats()


# Register router
app.include_router(router)


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    """Run the FastAPI application using uvicorn."""
    uvicorn.run(app, host="0.0.0.0", port=8000)
