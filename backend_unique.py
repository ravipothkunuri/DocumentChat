"""
DocumentChat Backend - Event-Driven RAG System with Plugin Architecture
Unique implementation using Event Bus, Registry Pattern, and Query Pipeline
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

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

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
FIXED_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
MAX_FILE_SIZE_MB = 20

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# EVENT BUS SYSTEM - Custom pub/sub for document lifecycle
# ============================================================================

class EventType(Enum):
    """Document lifecycle events"""
    DOC_UPLOADED = "doc_uploaded"
    DOC_PROCESSED = "doc_processed"
    DOC_DELETED = "doc_deleted"
    DOC_QUERIED = "doc_queried"
    SYSTEM_ERROR = "system_error"

@dataclass
class Event:
    """Event data container"""
    type: EventType
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class EventBus:
    """Central event dispatcher for system-wide notifications"""
    
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        self._listeners.setdefault(event_type, []).append(handler)
    
    def publish(self, event: Event):
        """Dispatch event to all subscribers"""
        for handler in self._listeners.get(event.type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")

# Global event bus
event_bus = EventBus()

# ============================================================================
# DOCUMENT LOADER REGISTRY - Dynamic loader registration pattern
# ============================================================================

class DocumentLoaderRegistry:
    """Plugin-style registry for document loaders"""
    
    def __init__(self):
        self._loaders: Dict[str, type] = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register built-in loaders"""
        self.register('.pdf', PyMuPDFLoader)
        self.register('.txt', TextLoader)
        self.register('.docx', Docx2txtLoader)
    
    def register(self, extension: str, loader_class: type):
        """Register new loader for file extension"""
        self._loaders[extension.lower()] = loader_class
        logger.info(f"Registered loader for {extension}")
    
    def get_loader(self, extension: str):
        """Get loader for file extension"""
        if extension.lower() not in self._loaders:
            raise ValueError(f"No loader registered for {extension}")
        return self._loaders[extension.lower()]
    
    def supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self._loaders.keys())

# Global loader registry
loader_registry = DocumentLoaderRegistry()

# ============================================================================
# CHUNKING STRATEGY PATTERN - Pluggable text splitting algorithms
# ============================================================================

class ChunkingStrategy:
    """Base class for text chunking strategies"""
    
    def chunk(self, text: str, filename: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

class RecursiveChunkingStrategy(ChunkingStrategy):
    """Recursive character-based chunking"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, filename: str) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        return [
            {
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "strategy": "recursive",
                    "size": len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]

class ChunkingStrategyFactory:
    """Factory for creating chunking strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: str = "recursive", **kwargs) -> ChunkingStrategy:
        """Create chunking strategy by type"""
        if strategy_type == "recursive":
            return RecursiveChunkingStrategy(**kwargs)
        raise ValueError(f"Unknown strategy: {strategy_type}")

# ============================================================================
# SEMANTIC INDEX - Custom vector store with advanced features
# ============================================================================

class SemanticIndex:
    """Vector store with metadata filtering and usage tracking"""
    
    def __init__(self, storage_path: Path = VECTOR_DIR / "semantic_index.json"):
        self.storage_path = storage_path
        self._chunks: List[Dict[str, Any]] = []
        self._vectors: Optional[np.ndarray] = None
        self._dimension: Optional[int] = None
        self._access_count: Dict[str, int] = {}  # Track chunk access frequency
        self.last_modified = None
    
    def index_chunks(self, chunks: List[Dict], vectors: List[List[float]]):
        """Index chunks with their embeddings"""
        if not chunks or not vectors or len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must match")
        
        vec_array = np.array(vectors, dtype=np.float32)
        
        if self._dimension is None:
            self._dimension = vec_array.shape[1]
        elif vec_array.shape[1] != self._dimension:
            raise ValueError(f"Vector dimension mismatch: {self._dimension} vs {vec_array.shape[1]}")
        
        self._chunks.extend(chunks)
        self._vectors = vec_array if self._vectors is None else np.vstack([self._vectors, vec_array])
        self.last_modified = datetime.now().isoformat()
    
    def search_similar(self, query_vector: List[float], top_k: int = 4, 
                       source_filter: Optional[str] = None) -> List[Tuple[Dict, float]]:
        """Search with optional metadata filtering"""
        if self._vectors is None or not self._chunks:
            return []
        
        # Convert to numpy
        query = np.array([query_vector], dtype=np.float32)
        
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        vectors_norm = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(query_norm, vectors_norm.T)[0]
        
        # Apply source filter if specified
        if source_filter:
            filtered_indices = [
                i for i, chunk in enumerate(self._chunks)
                if chunk.get('metadata', {}).get('source') == source_filter
            ]
            if not filtered_indices:
                return []
            filtered_sims = [(idx, similarities[idx]) for idx in filtered_indices]
            top_indices = sorted(filtered_sims, key=lambda x: x[1], reverse=True)[:top_k]
        else:
            top_k = min(top_k, len(self._chunks))
            top_indices = [(idx, similarities[idx]) for idx in np.argsort(similarities)[::-1][:top_k]]
        
        # Track access and return results
        results = []
        for idx, score in top_indices:
            chunk = self._chunks[idx]
            chunk_id = f"{chunk.get('metadata', {}).get('source')}:{chunk.get('metadata', {}).get('chunk_id')}"
            self._access_count[chunk_id] = self._access_count.get(chunk_id, 0) + 1
            results.append((chunk, float(score)))
        
        return results
    
    def remove_by_source(self, source: str):
        """Remove all chunks from a document"""
        indices_to_remove = [
            i for i, chunk in enumerate(self._chunks)
            if chunk.get('metadata', {}).get('source') == source
        ]
        
        if not indices_to_remove:
            return
        
        for idx in reversed(indices_to_remove):
            del self._chunks[idx]
        
        if self._vectors is not None:
            mask = np.ones(len(self._vectors), dtype=bool)
            mask[indices_to_remove] = False
            self._vectors = self._vectors[mask] if mask.any() else None
            self._dimension = None if self._vectors is None else self._dimension
        
        self.last_modified = datetime.now().isoformat()
    
    def get_chunk_heat(self, source: str) -> Dict[int, int]:
        """Get access frequency for chunks in a document"""
        heat_map = {}
        for chunk_id, count in self._access_count.items():
            if chunk_id.startswith(f"{source}:"):
                chunk_num = int(chunk_id.split(':')[1])
                heat_map[chunk_num] = count
        return heat_map
    
    def persist(self):
        """Save index to disk"""
        data = {
            "chunks": self._chunks,
            "vectors": self._vectors.tolist() if self._vectors is not None else None,
            "dimension": self._dimension,
            "access_count": self._access_count,
            "last_modified": self.last_modified
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def restore(self):
        """Load index from disk"""
        if not self.storage_path.exists():
            return
        
        data = json.loads(self.storage_path.read_text())
        self._chunks = data.get("chunks", [])
        self._dimension = data.get("dimension")
        self._access_count = data.get("access_count", {})
        self.last_modified = data.get("last_modified")
        
        if vectors := data.get("vectors"):
            self._vectors = np.array(vectors, dtype=np.float32)
        
        logger.info(f"Restored index with {len(self._chunks)} chunks")

# ============================================================================
# QUERY PIPELINE - Interceptor pattern for query processing
# ============================================================================

class QueryContext:
    """Context passed through query pipeline"""
    
    def __init__(self, question: str, top_k: int = 4):
        self.original_question = question
        self.processed_question = question
        self.retrieved_chunks: List[Tuple[Dict, float]] = []
        self.context_text = ""
        self.metadata: Dict[str, Any] = {}

class QueryInterceptor:
    """Base interceptor for query pipeline"""
    
    def process(self, context: QueryContext) -> QueryContext:
        raise NotImplementedError

class QueryEnhancementInterceptor(QueryInterceptor):
    """Enhance query with additional context"""
    
    def process(self, context: QueryContext) -> QueryContext:
        # Simple enhancement: add question marks if missing
        if not context.processed_question.endswith('?'):
            context.processed_question = f"{context.processed_question}?"
        return context

class RetrievalInterceptor(QueryInterceptor):
    """Retrieve relevant chunks"""
    
    def __init__(self, index: SemanticIndex, embedder):
        self.index = index
        self.embedder = embedder
    
    def process(self, context: QueryContext) -> QueryContext:
        query_vector = self.embedder.embed_query(context.processed_question)
        context.retrieved_chunks = self.index.search_similar(query_vector, context.metadata.get('top_k', 4))
        return context

class ContextBuilderInterceptor(QueryInterceptor):
    """Build context from retrieved chunks"""
    
    def process(self, context: QueryContext) -> QueryContext:
        if not context.retrieved_chunks:
            return context
        
        context_parts = []
        sources = []
        scores = []
        
        for chunk, score in context.retrieved_chunks:
            source = chunk.get('metadata', {}).get('source', 'unknown')
            content = chunk.get('content', '')
            context_parts.append(f"[Source: {source}]\n{content}")
            sources.append(source)
            scores.append(score)
        
        context.context_text = "\n\n---\n\n".join(context_parts)
        context.metadata['sources'] = sources
        context.metadata['scores'] = scores
        return context

class QueryPipeline:
    """Chain of interceptors for query processing"""
    
    def __init__(self):
        self.interceptors: List[QueryInterceptor] = []
    
    def add_interceptor(self, interceptor: QueryInterceptor):
        """Add interceptor to pipeline"""
        self.interceptors.append(interceptor)
        return self
    
    def execute(self, context: QueryContext) -> QueryContext:
        """Execute pipeline"""
        for interceptor in self.interceptors:
            context = interceptor.process(context)
        return context

# ============================================================================
# PROMPT TEMPLATE ENGINE
# ============================================================================

class PromptTemplate:
    """Template for generating prompts"""
    
    def __init__(self, template: str):
        self.template = template
    
    def render(self, **kwargs) -> str:
        """Render template with variables"""
        return self.template.format(**kwargs)

class PromptTemplateLibrary:
    """Collection of prompt templates"""
    
    DEFAULT_TEMPLATE = """You are a knowledgeable assistant with access to document content.

Available Information:
{context}

Question: {question}

Provide a helpful and accurate response based solely on the information provided above. If the information doesn't contain the answer, acknowledge this clearly."""
    
    FIRST_PERSON_TEMPLATE = """You are {doc_identity}, speaking in first person.

Your Content:
{context}

Question: {question}

Response (as {doc_identity}):"""
    
    @classmethod
    def get_template(cls, template_name: str = "default") -> PromptTemplate:
        """Get template by name"""
        templates = {
            "default": cls.DEFAULT_TEMPLATE,
            "first_person": cls.FIRST_PERSON_TEMPLATE
        }
        return PromptTemplate(templates.get(template_name, cls.DEFAULT_TEMPLATE))

# ============================================================================
# LLM STREAMING CLIENT
# ============================================================================

async def stream_llm_completion(prompt: str, model: str = FIXED_MODEL, 
                                temperature: float = 0.7) -> AsyncIterator[str]:
    """Stream LLM response"""
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            url = f"{OLLAMA_BASE_URL}/api/chat"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"temperature": temperature}
            }
            
            async with client.stream('POST', url, json=payload) as response:
                if response.status_code != 200:
                    raise ValueError(f"LLM service returned {response.status_code}")
                
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
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise ValueError(f"LLM error: {str(e)}")

# ============================================================================
# GLOBAL STATE
# ============================================================================

config_store = json.loads((VECTOR_DIR / "config.json").read_text()) if (VECTOR_DIR / "config.json").exists() else {
    'embedding_model': DEFAULT_EMBEDDING_MODEL,
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'temperature': 0.7,
    'total_queries': 0
}

metadata_store = json.loads((VECTOR_DIR / "metadata.json").read_text()) if (VECTOR_DIR / "metadata.json").exists() else {}

semantic_index = SemanticIndex()
embeddings_provider = None

def get_embeddings_provider():
    global embeddings_provider
    if embeddings_provider is None:
        embeddings_provider = OllamaEmbeddings(
            model=config_store['embedding_model'],
            base_url=OLLAMA_BASE_URL
        )
    return embeddings_provider

try:
    semantic_index.restore()
except Exception as e:
    logger.warning(f"Could not restore index: {e}")

# Setup event listeners
def log_event(event: Event):
    logger.info(f"Event: {event.type.value} at {event.timestamp}")

event_bus.subscribe(EventType.DOC_UPLOADED, log_event)
event_bus.subscribe(EventType.DOC_PROCESSED, log_event)
event_bus.subscribe(EventType.DOC_DELETED, log_event)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = None
    top_k: int = Field(4, ge=1, le=20)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False

class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str

class DocumentInfo(BaseModel):
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 DocumentChat system starting")
    yield
    logger.info("📻 Shutting down gracefully")

app = FastAPI(
    title="DocumentChat - Event-Driven RAG System",
    version="2.0.0",
    description="Unique implementation with Event Bus and Query Pipeline",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
                   allow_methods=["*"], allow_headers=["*"])

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_ok = response.status_code == 200
    except:
        ollama_ok = False
    
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ollama": "available" if ollama_ok else "unavailable",
            "semantic_index": "ready" if semantic_index._vectors is not None else "empty"
        },
        "statistics": {
            "documents": len(metadata_store),
            "indexed_chunks": len(semantic_index._chunks),
            "queries_served": config_store['total_queries']
        }
    }

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    
    # Validation
    extension = Path(file.filename).suffix.lower()
    if extension not in loader_registry.supported_extensions():
        raise HTTPException(400, f"Unsupported file type: {extension}")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    if file.filename in metadata_store:
        raise HTTPException(400, "Document already exists")
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Publish upload event
        event_bus.publish(Event(EventType.DOC_UPLOADED, {"filename": file.filename}))
        
        # Save file
        file_path.write_bytes(content)
        
        # Load document using registry
        loader_class = loader_registry.get_loader(extension)
        loader = loader_class(str(file_path))
        pages = [doc.page_content for doc in loader.load()]
        full_text = "\n\n".join(pages)
        
        # Chunk using strategy
        chunking_strategy = ChunkingStrategyFactory.create_strategy(
            "recursive",
            chunk_size=config_store['chunk_size'],
            overlap=config_store['chunk_overlap']
        )
        chunks = chunking_strategy.chunk(full_text, file.filename)
        
        if not chunks:
            raise ValueError("No content extracted")
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in chunks]
        vectors = get_embeddings_provider().embed_documents(texts)
        
        # Index chunks
        semantic_index.index_chunks(chunks, vectors)
        
        # Store metadata
        metadata_store[file.filename] = {
            "filename": file.filename,
            "size": len(content),
            "chunks": len(chunks),
            "status": "indexed",
            "uploaded_at": datetime.now().isoformat(),
            "type": extension[1:]
        }
        
        # Persist
        (VECTOR_DIR / "metadata.json").write_text(json.dumps(metadata_store, indent=2))
        semantic_index.persist()
        
        # Publish processed event
        event_bus.publish(Event(EventType.DOC_PROCESSED, {
            "filename": file.filename,
            "chunks": len(chunks)
        }))
        
        logger.info(f"Indexed {file.filename}: {len(chunks)} chunks")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(chunks),
            file_size=len(content),
            message=f"Indexed with {len(chunks)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if file_path.exists():
            file_path.unlink()
        metadata_store.pop(file.filename, None)
        event_bus.publish(Event(EventType.SYSTEM_ERROR, {"error": str(e)}))
        raise HTTPException(500, f"Upload failed: {str(e)}")

@router.post("/query")
async def query_documents(request: Request, query: QueryRequest):
    start_time = datetime.now()
    
    try:
        if not semantic_index._chunks:
            raise HTTPException(400, "No documents indexed")
        
        # Build query pipeline
        pipeline = QueryPipeline()
        pipeline.add_interceptor(QueryEnhancementInterceptor())
        pipeline.add_interceptor(RetrievalInterceptor(semantic_index, get_embeddings_provider()))
        pipeline.add_interceptor(ContextBuilderInterceptor())
        
        # Execute pipeline
        context = QueryContext(query.question)
        context.metadata['top_k'] = query.top_k
        context = pipeline.execute(context)
        
        if not context.retrieved_chunks:
            raise HTTPException(400, "No relevant content found")
        
        # Build prompt using template
        unique_sources = list(set(context.metadata['sources']))
        doc_identity = unique_sources[0] if len(unique_sources) == 1 else "your documents"
        
        template = PromptTemplateLibrary.get_template("first_person")
        prompt = template.render(
            doc_identity=doc_identity,
            context=context.context_text,
            question=context.processed_question
        )
        
        # Publish query event
        event_bus.publish(Event(EventType.DOC_QUERIED, {
            "question": query.question,
            "sources": unique_sources
        }))
        
        # Stream response
        if query.stream:
            async def generate():
                try:
                    # Send metadata
                    yield f"data: {json.dumps({'type': 'metadata', 'sources': context.metadata['sources'], 'chunks_used': len(context.retrieved_chunks), 'similarity_scores': context.metadata['scores'], 'model_used': FIXED_MODEL})}\n\n"
                    
                    # Stream content
                    async for chunk in stream_llm_completion(prompt, FIXED_MODEL, query.temperature):
                        if await request.is_disconnected():
                            break
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                    
                    # Send completion
                    elapsed = (datetime.now() - start_time).total_seconds()
                    yield f"data: {json.dumps({'type': 'done', 'processing_time': elapsed})}\n\n"
                    
                    config_store['total_queries'] += 1
                    (VECTOR_DIR / "config.json").write_text(json.dumps(config_store, indent=2))
                    
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Non-streaming
        else:
            full_response = ""
            async for chunk in stream_llm_completion(prompt, FIXED_MODEL, query.temperature):
                full_response += chunk
            
            elapsed = (datetime.now() - start_time).total_seconds()
            config_store['total_queries'] += 1
            (VECTOR_DIR / "config.json").write_text(json.dumps(config_store, indent=2))
            
            return {
                "answer": full_response,
                "sources": context.metadata['sources'],
                "chunks_used": len(context.retrieved_chunks),
                "similarity_scores": context.metadata['scores'],
                "processing_time": elapsed,
                "model_used": FIXED_MODEL
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    return [DocumentInfo(**meta) for meta in metadata_store.values()]

@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    if filename not in metadata_store:
        raise HTTPException(404, "Document not found")
    
    try:
        semantic_index.remove_by_source(filename)
        metadata_store.pop(filename)
        (VECTOR_DIR / "metadata.json").write_text(json.dumps(metadata_store, indent=2))
        
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        semantic_index.persist()
        
        event_bus.publish(Event(EventType.DOC_DELETED, {"filename": filename}))
        
        return {"status": "success", "message": f"Deleted {filename}"}
        
    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")

@router.delete("/clear")
async def clear_all_documents():
    try:
        semantic_index._chunks = []
        semantic_index._vectors = None
        semantic_index._dimension = None
        semantic_index._access_count = {}
        semantic_index.last_modified = datetime.now().isoformat()
        
        metadata_store.clear()
        (VECTOR_DIR / "metadata.json").write_text(json.dumps(metadata_store, indent=2))
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        semantic_index.persist()
        return {"status": "success", "message": "All documents cleared"}
        
    except Exception as e:
        raise HTTPException(500, f"Clear failed: {str(e)}")

@router.get("/stats")
async def get_statistics():
    doc_count = len(metadata_store)
    total_size = sum(meta.get('size', 0) for meta in metadata_store.values())
    
    return {
        "total_documents": doc_count,
        "total_chunks": len(semantic_index._chunks),
        "total_queries": config_store['total_queries'],
        "total_storage_size": total_size,
        "average_chunks_per_document": round(len(semantic_index._chunks) / max(1, doc_count), 2),
        "last_update": semantic_index.last_modified
    }

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
