import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama

from vector_store import VectorStore

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Add request logging middleware
class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            logger.debug(f"[REQUEST] {scope['method']} {scope['path']}")
        await self.app(scope, receive, send)

# Pydantic Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
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

class DocumentInfo(BaseModel):
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str

# Global configuration
class Config:
    def __init__(self):
        self.model = "llama3"
        self.embedding_model = "nomic-embed-text"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.temperature = 0.7
        self.total_queries = 0
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        logger.info(f"[CONFIG] Ollama base URL: {self.ollama_base_url}")

config = Config()

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Ollama RAG Assistant API",
    description="Production-ready RAG system powered by LangChain, Ollama local LLMs, and custom vector store with Streamlit UI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_store = VectorStore()
embeddings_model = None
llm_model = None
document_metadata = {}

# Create directories
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

METADATA_FILE = VECTOR_DIR / "metadata.json"
CONFIG_FILE = VECTOR_DIR / "config.json"
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def check_ollama_connection():
    """Check if Ollama server is accessible."""
    import requests
    try:
        logger.debug(f"[OLLAMA CHECK] Testing connection to {config.ollama_base_url}")
        response = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"[OLLAMA CHECK] ✓ Connected successfully. Found {len(models)} models")
            for model in models:
                logger.debug(f"[OLLAMA CHECK]   - {model.get('name', 'unknown')}")
            return True, models
        else:
            logger.error(f"[OLLAMA CHECK] ✗ Unexpected status code: {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[OLLAMA CHECK] ✗ Connection failed: {e}")
        logger.error(f"[OLLAMA CHECK] Make sure Ollama is running: 'ollama serve'")
        return False, []
    except Exception as e:
        logger.error(f"[OLLAMA CHECK] ✗ Error: {e}")
        return False, []

def load_config():
    """Load configuration from file."""
    try:
        if CONFIG_FILE.exists():
            logger.debug(f"[CONFIG] Loading from {CONFIG_FILE}")
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
            config.model = config_data.get('model', 'llama3')
            config.embedding_model = config_data.get('embedding_model', 'nomic-embed-text')
            config.chunk_size = config_data.get('chunk_size', 1000)
            config.chunk_overlap = config_data.get('chunk_overlap', 200)
            config.temperature = config_data.get('temperature', 0.7)
            config.total_queries = config_data.get('total_queries', 0)
            logger.info(f"[CONFIG] ✓ Loaded: model={config.model}, embedding={config.embedding_model}")
        else:
            logger.info("[CONFIG] No config file found, using defaults")
    except Exception as e:
        logger.error(f"[CONFIG] ✗ Error loading config: {e}")

def save_config():
    """Save configuration to file."""
    try:
        config_data = {
            'model': config.model,
            'embedding_model': config.embedding_model,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'temperature': config.temperature,
            'total_queries': config.total_queries
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.debug("[CONFIG] ✓ Saved successfully")
    except Exception as e:
        logger.error(f"[CONFIG] ✗ Error saving: {e}")

def load_metadata():
    """Load document metadata from file."""
    global document_metadata
    try:
        if METADATA_FILE.exists():
            logger.debug(f"[METADATA] Loading from {METADATA_FILE}")
            with open(METADATA_FILE, 'r') as f:
                document_metadata = json.load(f)
            logger.info(f"[METADATA] ✓ Loaded {len(document_metadata)} documents")
        else:
            document_metadata = {}
            logger.info("[METADATA] No metadata file found")
    except Exception as e:
        logger.error(f"[METADATA] ✗ Error loading: {e}")
        document_metadata = {}

def save_metadata():
    """Save document metadata to file."""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(document_metadata, f, indent=2)
        logger.debug("[METADATA] ✓ Saved successfully")
    except Exception as e:
        logger.error(f"[METADATA] ✗ Error saving: {e}")

def get_embeddings_model():
    """Get or create embeddings model (LangChain)."""
    global embeddings_model
    try:
        if embeddings_model is None:
            logger.info(f"[EMBEDDINGS] Initializing model: {config.embedding_model}")
            logger.debug(f"[EMBEDDINGS] Base URL: {config.ollama_base_url}")
            
            embeddings_model = OllamaEmbeddings(
                model=config.embedding_model,
                base_url=config.ollama_base_url
            )
            
            # Test the model
            logger.debug("[EMBEDDINGS] Testing with sample query...")
            test_embedding = embeddings_model.embed_query("test")
            logger.info(f"[EMBEDDINGS] ✓ Initialized successfully, dimensions: {len(test_embedding)}")
        
        return embeddings_model
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[EMBEDDINGS] ✗ Failed to initialize '{config.embedding_model}': {error_msg}")
        logger.debug(f"[EMBEDDINGS] Traceback:\n{traceback.format_exc()}")
        
        # Check if it's a model not found error
        if "404" in error_msg or "not found" in error_msg.lower():
            logger.error(f"[EMBEDDINGS] Model not found. Try: ollama pull {config.embedding_model}")
            raise HTTPException(
                status_code=404,
                detail=f"Embedding model '{config.embedding_model}' not found. Please pull it first using: ollama pull {config.embedding_model}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize embeddings model '{config.embedding_model}': {error_msg}"
            )

def get_llm_model(model_name: str = None):
    """Get or create LLM model (LangChain)."""
    global llm_model
    try:
        model_to_use = model_name or config.model
        
        if llm_model is None or llm_model.model != model_to_use:
            logger.info(f"[LLM] Initializing model: {model_to_use}")
            logger.debug(f"[LLM] Base URL: {config.ollama_base_url}")
            logger.debug(f"[LLM] Temperature: {config.temperature}")
            
            llm_model = ChatOllama(
                model=model_to_use,
                temperature=config.temperature,
                base_url=config.ollama_base_url
            )
            
            logger.info(f"[LLM] ✓ Initialized successfully: {model_to_use}")
        
        return llm_model
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[LLM] ✗ Failed to initialize '{model_to_use}': {error_msg}")
        logger.debug(f"[LLM] Traceback:\n{traceback.format_exc()}")
        
        # Check if it's a model not found error
        if "404" in error_msg or "not found" in error_msg.lower():
            logger.error(f"[LLM] Model not found. Try: ollama pull {model_to_use}")
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_to_use}' not found. Please pull it first using: ollama pull {model_to_use}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize LLM model '{model_to_use}': {error_msg}"
            )

def validate_file_type(filename: str) -> bool:
    """Validate file type based on extension."""
    is_valid = Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
    logger.debug(f"[VALIDATION] File type check for '{filename}': {is_valid}")
    return is_valid

def load_document(file_path: Path) -> List[str]:
    """Load document content based on file type (LangChain)."""
    try:
        suffix = file_path.suffix.lower()
        logger.debug(f"[LOADER] Loading {suffix} file: {file_path.name}")
        
        if suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            content = [doc.page_content for doc in documents]
        elif suffix == '.txt':
            loader = TextLoader(str(file_path))
            documents = loader.load()
            content = [doc.page_content for doc in documents]
        elif suffix == '.docx':
            loader = UnstructuredWordDocumentLoader(str(file_path))
            documents = loader.load()
            content = [doc.page_content for doc in documents]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        logger.info(f"[LOADER] ✓ Loaded {len(content)} pages from {file_path.name}")
        return content
        
    except Exception as e:
        logger.error(f"[LOADER] ✗ Error loading {file_path}: {e}")
        raise

def split_text(content: List[str], filename: str) -> List[Dict[str, Any]]:
    """Split text content into chunks (LangChain)."""
    try:
        logger.debug(f"[SPLITTER] Processing {filename} with {len(content)} pages")
        
        # Combine all pages
        full_text = "\n\n".join(content)
        logger.debug(f"[SPLITTER] Total text length: {len(full_text)} characters")
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split text
        chunks = text_splitter.split_text(full_text)
        
        # Create document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "page_content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "chunk_length": len(chunk)
                }
            }
            documents.append(doc)
        
        logger.info(f"[SPLITTER] ✓ Split {filename} into {len(documents)} chunks")
        return documents
        
    except Exception as e:
        logger.error(f"[SPLITTER] ✗ Error splitting {filename}: {e}")
        raise

def clean_llm_response(text: str) -> str:
    """Clean LLM response by removing reasoning tags."""
    import re
    
    logger.debug("[CLEANER] Cleaning LLM response")
    
    # Remove <think>...</think> tags and content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any standalone opening/closing tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?reasoning>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?thought>', '', text, flags=re.IGNORECASE)
    
    # Clean up excessive whitespace/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text

def check_ollama_available():
    """Check if Ollama is available and models are accessible."""
    try:
        logger.debug("[HEALTH] Checking Ollama availability")
        
        # Try to initialize embeddings model
        embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url
        )
        test_embedding = embeddings.embed_query("test")
        
        # Try to initialize LLM
        llm = ChatOllama(
            model=config.model,
            base_url=config.ollama_base_url
        )
        
        logger.info(f"[HEALTH] ✓ Ollama available with {config.embedding_model} and {config.model}")
        return True, f"Ollama available with {config.embedding_model} and {config.model}"
    except Exception as e:
        logger.error(f"[HEALTH] ✗ Ollama check failed: {e}")
        return False, str(e)

def get_available_models():
    """Get available models from Ollama."""
    try:
        import subprocess
        logger.debug("[MODELS] Getting available models from Ollama")
        
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name.split(':')[0])  # Remove tag
            
            # Filter out embedding models from LLM list
            embedding_models = [m for m in models if 'embed' in m.lower() or 'nomic' in m.lower()]
            llm_models = [m for m in models if 'embed' not in m.lower()]
            
            logger.info(f"[MODELS] ✓ Found {len(llm_models)} LLM models, {len(embedding_models)} embedding models")
            return llm_models, embedding_models
        else:
            logger.warning(f"[MODELS] ✗ Command failed: {result.stderr}")
            return [], []
            
    except Exception as e:
        logger.error(f"[MODELS] ✗ Error: {e}")
        return [], []

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("="*80)
    logger.info("[STARTUP] Starting RAG Application...")
    logger.info("="*80)
    
    # Check Ollama connection first
    connected, models = check_ollama_connection()
    if not connected:
        logger.warning("[STARTUP] ⚠ Ollama not accessible - some features may not work")
    
    # Load configuration
    load_config()
    
    # Load existing data
    load_metadata()
    
    try:
        vector_store.load()
        logger.info("[STARTUP] ✓ Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"[STARTUP] ⚠ Could not load vector store: {e}")
    
    logger.info("="*80)
    logger.info("[STARTUP] ✓ RAG Application started successfully")
    logger.info(f"[STARTUP] Documents: {len(document_metadata)}, Vector chunks: {vector_store.get_stats().get('total_chunks', 0)}")
    logger.info("="*80)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Check system health and configuration."""
    logger.debug("[ENDPOINT] /health called")
    
    try:
        # Check Ollama availability
        ollama_available, ollama_message = check_ollama_available()
        
        # Get statistics
        stats = vector_store.get_stats()
        
        response = {
            "status": "healthy" if ollama_available else "degraded",
            "timestamp": datetime.now().isoformat(),
            "ollama_status": {
                "available": ollama_available,
                "message": ollama_message,
                "base_url": config.ollama_base_url
            },
            "configuration": {
                "model": config.model,
                "embedding_model": config.embedding_model,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "temperature": config.temperature
            },
            "document_count": len(document_metadata),
            "total_chunks": stats.get("total_chunks", 0),
            "total_queries": config.total_queries
        }
        
        logger.info(f"[ENDPOINT] /health completed: {response['status']}")
        return response
        
    except Exception as e:
        logger.error(f"[ENDPOINT] /health failed: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Upload endpoint
@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document synchronously."""
    logger.info(f"[ENDPOINT] /upload called for: {file.filename}")
    
    # Validate file type
    if not validate_file_type(file.filename):
        logger.warning(f"[UPLOAD] ✗ Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check for duplicates
    if file.filename in document_metadata:
        logger.warning(f"[UPLOAD] ✗ Duplicate file: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists. Please delete it first or rename your file."
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save uploaded file
        logger.debug(f"[UPLOAD] Saving to {file_path}")
        content = await file.read()
        file_size = len(content)
        
        # Check file size
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.warning(f"[UPLOAD] ✗ File too large: {file_size} bytes")
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB"
            )
        
        with open(file_path, 'wb') as f:
            f.write(content)
        logger.debug(f"[UPLOAD] ✓ File saved: {file_size} bytes")
        
        # Load and process document
        document_content = load_document(file_path)
        
        # Split into chunks
        documents = split_text(document_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        logger.debug(f"[UPLOAD] Generating embeddings for {len(documents)} chunks")
        embeddings_model = get_embeddings_model()
        
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        logger.info(f"[UPLOAD] ✓ Generated {len(embeddings)} embeddings")
        
        # Add to vector store
        logger.debug(f"[UPLOAD] Adding to vector store")
        vector_store.add_documents(documents, embeddings)
        
        # Save metadata
        document_metadata[file.filename] = {
            "filename": file.filename,
            "size": file_size,
            "chunks": len(documents),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        }
        
        save_metadata()
        vector_store.save()
        
        logger.info(f"[UPLOAD] ✓ Successfully processed {file.filename}: {len(documents)} chunks, {file_size} bytes")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=file_size,
            message=f"Document processed successfully into {len(documents)} chunks"
        )
        
    except Exception as e:
        logger.error(f"[UPLOAD] ✗ Error processing {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()
        
        if file.filename in document_metadata:
            del document_metadata[file.filename]
            save_metadata()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

# Query endpoint
@app.post("/query", tags=["Query"])
async def query_documents(request: QueryRequest):
    """Query the document collection."""
    logger.info(f"[ENDPOINT] /query called: '{request.question[:100]}...'")
    logger.debug(f"[QUERY] Parameters: model={request.model}, top_k={request.top_k}, stream={request.stream}, temp={request.temperature}")
    
    start_time = datetime.now()
    
    try:
        # Check if vector store has documents
        stats = vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            logger.warning("[QUERY] ✗ No documents available")
            raise HTTPException(
                status_code=400,
                detail="No documents available for querying. Please upload some documents first."
            )
        
        # Generate query embedding
        logger.debug("[QUERY] Generating query embedding")
        embeddings_model = get_embeddings_model()
        query_embedding = embeddings_model.embed_query(request.question)
        logger.debug(f"[QUERY] ✓ Embedding generated: {len(query_embedding)} dimensions")
        
        # Perform similarity search
        logger.debug(f"[QUERY] Performing similarity search (top_k={request.top_k})")
        similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
        if not similar_docs:
            logger.warning("[QUERY] ✗ No relevant documents found")
            raise HTTPException(
                status_code=400,
                detail="No relevant documents found for your query."
            )
        
        logger.info(f"[QUERY] ✓ Found {len(similar_docs)} relevant chunks")
        for i, (doc, score) in enumerate(similar_docs):
            logger.debug(f"[QUERY]   {i+1}. {doc['metadata']['source']} (score: {score:.4f})")
        
        # Build context
        context_parts = []
        sources = []
        similarity_scores = []
        
        for doc, score in similar_docs:
            context_parts.append(f"Source: {doc['metadata']['source']}\nContent: {doc['page_content']}")
            sources.append(doc['metadata']['source'])
            similarity_scores.append(float(score))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using LLM
        logger.debug(f"[QUERY] Generating answer with LLM")
        llm = get_llm_model(request.model)
        
        if request.temperature is not None:
            llm.temperature = request.temperature
        
        # Get unique source filenames for personalization
        unique_sources = list(set(sources))
        doc_identity = unique_sources[0] if len(unique_sources) == 1 else f"your documents ({', '.join(unique_sources[:2])}{'...' if len(unique_sources) > 2 else ''})"
        
        prompt = f"""You are {doc_identity}, a helpful document assistant. You should respond in first person as if you are the document itself, speaking directly to the user.

Your content includes:
{context}

The user asks: {request.question}

Respond naturally and conversationally as the document, using "I" when referring to your content. For example:
- "Based on what I contain, I can tell you that..."
- "In my section about X, I mention that..."
- "I don't seem to have information about that topic."

Your response:"""
        
        # Stream or regular response
        if request.stream:
            logger.debug("[QUERY] Starting streaming response")
            
            async def generate():
                # Send metadata first
                metadata = {
                    "sources": sources,
                    "chunks_used": len(similar_docs),
                    "similarity_scores": similarity_scores,
                    "type": "metadata"
                }
                yield f"data: {json.dumps(metadata)}\n\n"
                
                # Accumulate full response
                full_answer = ""
                
                try:
                    for chunk in llm.stream(prompt):
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        full_answer += content
                except Exception as stream_error:
                    logger.error(f"[QUERY] ✗ Streaming error: {stream_error}")
                    logger.debug(traceback.format_exc())
                    yield f"data: {json.dumps({'type': 'error', 'content': str(stream_error)})}\n\n"
                    return
                
                # Clean the complete response
                cleaned_answer = clean_llm_response(full_answer)
                
                # Stream the cleaned answer
                chunk_size = 50
                for i in range(0, len(cleaned_answer), chunk_size):
                    chunk_text = cleaned_answer[i:i + chunk_size]
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk_text})}\n\n"
                
                # Send completion
                processing_time = (datetime.now() - start_time).total_seconds()
                completion = {
                    "type": "done",
                    "processing_time": processing_time
                }
                yield f"data: {json.dumps(completion)}\n\n"
                
                # Update query count
                config.total_queries += 1
                save_config()
                logger.info(f"[QUERY] ✓ Streaming completed in {processing_time:.2f}s")
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Regular response
            logger.debug("[QUERY] Generating regular response")
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Clean the answer before returning
            answer = clean_llm_response(answer)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update query count
            config.total_queries += 1
            save_config()
            
            logger.info(f"[QUERY] ✓ Completed in {processing_time:.2f}s, {len(similar_docs)} chunks used")
            
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
        logger.error(f"[QUERY] ✗ Error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

# List documents endpoint
@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all uploaded documents."""
    logger.debug("[ENDPOINT] /documents called")
    
    documents = []
    for filename, metadata in document_metadata.items():
        documents.append(DocumentInfo(**metadata))
    
    logger.info(f"[ENDPOINT] /documents returned {len(documents)} documents")
    return documents

# Delete document endpoint
@app.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a specific document."""
    logger.info(f"[ENDPOINT] /documents/{filename} DELETE called")
    
    if filename not in document_metadata:
        logger.warning(f"[DELETE] ✗ Document not found: {filename}")
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found"
        )
    
    try:
        # Remove from vector store
        logger.debug(f"[DELETE] Removing from vector store")
        vector_store.remove_documents_by_source(filename)
        
        # Remove metadata
        del document_metadata[filename]
        save_metadata()
        
        # Remove file
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"[DELETE] ✓ File removed")
        
        # Save vector store
        vector_store.save()
        
        logger.info(f"[DELETE] ✓ Successfully deleted: {filename}")
        return {
            "message": f"Document '{filename}' deleted successfully",
            "note": "Vector store has been updated automatically"
        }
        
    except Exception as e:
        logger.error(f"[DELETE] ✗ Error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

# Clear all documents endpoint
@app.delete("/clear", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents and embeddings."""
    logger.info("[ENDPOINT] /clear called")
    
    try:
        # Clear vector store
        vector_store.clear()
        
        # Clear metadata
        global document_metadata
        document_metadata = {}
        save_metadata()
        
        # Remove all uploaded files
        file_count = 0
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                file_count += 1
        
        # Save empty vector store
        vector_store.save()
        
        logger.info(f"[CLEAR] ✓ Cleared {file_count} files and all embeddings")
        return {
            "message": "All documents and embeddings cleared successfully",
            "cleared": True
        }
        
    except Exception as e:
        logger.error(f"[CLEAR] ✗ Error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear documents: {str(e)}"
        )

# Preview document endpoint
@app.get("/documents/{filename}/preview", tags=["Documents"])
async def preview_document(filename: str, num_chunks: int = Query(3, ge=1, le=10)):
    """Preview document chunks."""
    logger.debug(f"[ENDPOINT] /documents/{filename}/preview called (num_chunks={num_chunks})")
    
    if filename not in document_metadata:
        logger.warning(f"[PREVIEW] ✗ Document not found: {filename}")
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found"
        )
    
    try:
        # Get documents from vector store
        documents = vector_store.get_documents_by_source(filename)
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for document '{filename}'"
            )
        
        # Return first num_chunks
        preview_chunks = documents[:num_chunks]
        
        chunks_data = []
        for doc in preview_chunks:
            chunks_data.append({
                "chunk_id": doc['metadata']['chunk_id'],
                "content": doc['page_content'][:500] + ("..." if len(doc['page_content']) > 500 else ""),
                "length": len(doc['page_content'])
            })
        
        logger.info(f"[PREVIEW] ✓ Generated preview: {len(chunks_data)} chunks")
        
        return {
            "filename": filename,
            "total_chunks": len(documents),
            "preview_chunks": len(chunks_data),
            "chunks": chunks_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PREVIEW] ✗ Error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )

# Configuration endpoint
@app.post("/configure", tags=["Configuration"])
async def configure_system(config_update: ModelConfig):
    """Update system configuration."""
    logger.info(f"[ENDPOINT] /configure called")
    logger.debug(f"[CONFIG] Update request: {config_update.dict(exclude_none=True)}")
    
    changed_fields = []
    
    try:
        # Update configuration
        if config_update.model is not None:
            if config.model != config_update.model:
                logger.info(f"[CONFIG] Changing LLM model: {config.model} -> {config_update.model}")
                config.model = config_update.model
                changed_fields.append("model")
                global llm_model
                llm_model = None  # Reset to force reinitialization
        
        if config_update.embedding_model is not None:
            if config.embedding_model != config_update.embedding_model:
                logger.info(f"[CONFIG] Changing embedding model: {config.embedding_model} -> {config_update.embedding_model}")
                config.embedding_model = config_update.embedding_model
                changed_fields.append("embedding_model")
                global embeddings_model
                embeddings_model = None  # Reset to force reinitialization
        
        if config_update.chunk_size is not None:
            if config.chunk_size != config_update.chunk_size:
                logger.info(f"[CONFIG] Changing chunk_size: {config.chunk_size} -> {config_update.chunk_size}")
                config.chunk_size = config_update.chunk_size
                changed_fields.append("chunk_size")
        
        if config_update.chunk_overlap is not None:
            if config.chunk_overlap != config_update.chunk_overlap:
                logger.info(f"[CONFIG] Changing chunk_overlap: {config.chunk_overlap} -> {config_update.chunk_overlap}")
                config.chunk_overlap = config_update.chunk_overlap
                changed_fields.append("chunk_overlap")
        
        if config_update.temperature is not None:
            if config.temperature != config_update.temperature:
                logger.info(f"[CONFIG] Changing temperature: {config.temperature} -> {config_update.temperature}")
                config.temperature = config_update.temperature
                changed_fields.append("temperature")
                if llm_model:
                    llm_model.temperature = config.temperature
        
        # Save configuration to file
        save_config()
        
        logger.info(f"[CONFIG] ✓ Updated successfully, changed: {changed_fields}")
        
        response = {
            "message": "Configuration updated successfully",
            "changed_fields": changed_fields
        }
        
        if "embedding_model" in changed_fields:
            response["warning"] = "Embedding model changed. Consider rebuilding vectors for existing documents."
        
        return response
        
    except Exception as e:
        logger.error(f"[CONFIG] ✗ Error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )

# Get models endpoint
@app.get("/models", tags=["Configuration"])
async def get_models():
    """Get available models and current configuration."""
    logger.debug("[ENDPOINT] /models called")
    
    try:
        llm_models, embedding_models = get_available_models()
        
        # Fallback to default models if none found
        if not llm_models:
            llm_models = ["llama3", "mistral", "phi"]
        if not embedding_models:
            embedding_models = ["nomic-embed-text"]
        
        response = {
            "llm_models": llm_models,
            "embedding_models": embedding_models,
            "current_llm": config.model,
            "current_embedding": config.embedding_model
        }
        
        logger.info(f"[MODELS] ✓ Returned {len(llm_models)} LLM, {len(embedding_models)} embedding models")
        return response
        
    except Exception as e:
        logger.error(f"[MODELS] ✗ Error: {e}")
        # Return defaults on error
        return {
            "llm_models": ["llama3", "mistral", "phi"],
            "embedding_models": ["nomic-embed-text"],
            "current_llm": config.model,
            "current_embedding": config.embedding_model
        }

# Statistics endpoint
@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get system statistics."""
    logger.debug("[ENDPOINT] /stats called")
    
    try:
        stats = vector_store.get_stats()
        
        total_size = sum(meta.get('size', 0) for meta in document_metadata.values())
        avg_chunks = stats.get('total_chunks', 0) / max(1, len(document_metadata))
        
        return {
            "total_documents": len(document_metadata),
            "total_chunks": stats.get('total_chunks', 0),
            "total_queries": config.total_queries,
            "total_storage_size": total_size,
            "average_chunks_per_document": round(avg_chunks, 2),
            "last_update": stats.get('last_update')
        }
        
    except Exception as e:
        logger.error(f"[STATS] ✗ Error: {e}")
        return {
            "total_documents": len(document_metadata),
            "total_chunks": 0,
            "total_queries": config.total_queries,
            "total_storage_size": 0,
            "average_chunks_per_document": 0.0,
            "last_update": None
        }

# Debug embedding endpoint
@app.get("/debug/embeddings", tags=["Debug"])
async def debug_embeddings(text: str = Query("This is a test sentence")):
    """Test embedding generation."""
    logger.info(f"[ENDPOINT] /debug/embeddings called with: '{text[:50]}...'")
    
    try:
        start_time = datetime.now()
        embeddings_model = get_embeddings_model()
        embedding = embeddings_model.embed_query(text)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"[DEBUG] ✓ Embedding generated: {len(embedding)} dims in {generation_time:.3f}s")
        
        return {
            "status": "success",
            "model": config.embedding_model,
            "dimensions": len(embedding),
            "generation_time": generation_time,
            "embedding": embedding[:10] if embedding else []  # First 10 values for preview
        }
        
    except Exception as e:
        logger.error(f"[DEBUG] ✗ Error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {str(e)}"
        )

# Debug vector store endpoint
@app.get("/debug/vector-store", tags=["Debug"])
async def debug_vector_store():
    """Inspect vector store state."""
    logger.debug("[ENDPOINT] /debug/vector-store called")
    
    try:
        stats = vector_store.get_stats()
        sample_docs = vector_store.get_sample_documents(5)
        
        logger.info(f"[DEBUG] Vector store stats: {stats.get('total_chunks', 0)} chunks")
        
        return {
            "total_documents": len(document_metadata),
            "total_chunks": stats.get('total_chunks', 0),
            "embedding_dimensions": stats.get('embedding_dimensions'),
            "dimension_consistent": stats.get('dimension_consistent', True),
            "vector_store_size": stats.get('vector_store_size', 0),
            "sample_documents": [
                {
                    "source": doc.get('metadata', {}).get('source', 'Unknown'),
                    "chunk_id": doc.get('metadata', {}).get('chunk_id', 'N/A'),
                    "content_preview": doc.get('page_content', '')[:100] + "..." if len(doc.get('page_content', '')) > 100 else doc.get('page_content', '')
                }
                for doc in sample_docs
            ]
        }
        
    except Exception as e:
        logger.error(f"[DEBUG] ✗ Error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to inspect vector store: {str(e)}"
        )

# Test Ollama connection endpoint
@app.get("/debug/ollama", tags=["Debug"])
async def debug_ollama():
    """Test Ollama connection and list available models."""
    logger.info("[ENDPOINT] /debug/ollama called")
    
    connected, models = check_ollama_connection()
    
    return {
        "connected": connected,
        "base_url": config.ollama_base_url,
        "models": [m.get('name', 'unknown') for m in models] if models else [],
        "model_count": len(models) if models else 0,
        "current_llm": config.model,
        "current_embedding": config.embedding_model
    }

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("[MAIN] Starting RAG Backend Server on port 8000...")
    logger.info("="*80)
    uvicorn.run(
        "rag_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
