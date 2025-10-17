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
        # CHANGED: No hardcoded defaults - will be loaded from config file or auto-detected
        self.model = None
        self.embedding_model = None
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.temperature = 0.7
        self.total_queries = 0

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_store = VectorStore()
embeddings_model = None
llm_model = None
llm_model_name = None  # Track current model name
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

def load_config():
    """Load configuration from file."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
            config.model = config_data.get('model', 'phi3')  # FIXED: Changed default
            config.embedding_model = config_data.get('embedding_model', 'nomic-embed-text')
            config.chunk_size = config_data.get('chunk_size', 1000)
            config.chunk_overlap = config_data.get('chunk_overlap', 200)
            config.temperature = config_data.get('temperature', 0.7)
            config.total_queries = config_data.get('total_queries', 0)
            logger.info(f"Loaded configuration: model={config.model}, embedding={config.embedding_model}")
        else:
            logger.info("No existing config file found, using defaults")
    except Exception as e:
        logger.error(f"Error loading config: {e}")

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
        logger.debug("Configuration saved successfully")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def load_metadata():
    """Load document metadata from file."""
    global document_metadata
    try:
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'r') as f:
                document_metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(document_metadata)} documents")
        else:
            document_metadata = {}
            logger.info("No existing metadata file found")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        document_metadata = {}

def save_metadata():
    """Save document metadata to file."""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(document_metadata, f, indent=2)
        logger.debug("Metadata saved successfully")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")

def get_embeddings_model():
    """Get or create embeddings model (LangChain)."""
    global embeddings_model
    try:
        if embeddings_model is None:
            logger.info(f"Initializing embeddings model: {config.embedding_model}")
            embeddings_model = OllamaEmbeddings(
                model=config.embedding_model,
                base_url="http://localhost:11434"  # ADDED: Explicit base URL
            )
            
            # Test the model
            test_embedding = embeddings_model.embed_query("test")
            logger.info(f"Embeddings model initialized successfully, dimensions: {len(test_embedding)}")
        
        return embeddings_model
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error initializing embeddings model '{config.embedding_model}': {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        if "404" in error_msg or "not found" in error_msg.lower():
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
    global llm_model, llm_model_name
    try:
        model_to_use = model_name or config.model
        
        # FIXED: Better model caching logic
        if llm_model is None or llm_model_name != model_to_use:
            logger.info(f"Initializing LLM model: {model_to_use}")
            llm_model = ChatOllama(
                model=model_to_use,
                temperature=config.temperature,
                base_url="http://localhost:11434"  # ADDED: Explicit base URL
            )
            llm_model_name = model_to_use
            logger.info(f"LLM model '{model_to_use}' initialized successfully")
        
        return llm_model
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error initializing LLM model '{model_to_use}': {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        if "404" in error_msg or "not found" in error_msg.lower():
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
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def load_document(file_path: Path) -> List[str]:
    """Load document content based on file type (LangChain)."""
    try:
        suffix = file_path.suffix.lower()
        
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
        
        logger.debug(f"Loaded {len(content)} pages from {file_path}")
        return content
        
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        raise

def split_text(content: List[str], filename: str) -> List[Dict[str, Any]]:
    """Split text content into chunks (LangChain)."""
    try:
        full_text = "\n\n".join(content)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(full_text)
        
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
        
        logger.info(f"Split {filename} into {len(documents)} chunks")
        return documents
        
    except Exception as e:
        logger.error(f"Error splitting text for {filename}: {e}")
        raise

def clean_llm_response(text: str) -> str:
    """Clean LLM response by removing reasoning tags like <think>, <reasoning>, etc."""
    import re
    
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
        embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url="http://localhost:11434"
        )
        test_embedding = embeddings.embed_query("test")
        
        llm = ChatOllama(
            model=config.model,
            base_url="http://localhost:11434"
        )
        
        return True, f"Ollama available with {config.embedding_model} and {config.model}"
    except Exception as e:
        return False, str(e)

def get_available_models():
    """Get available models from Ollama."""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name.split(':')[0])
            
            embedding_models = [m for m in models if 'embed' in m.lower() or 'nomic' in m.lower()]
            llm_models = [m for m in models if 'embed' not in m.lower()]
            
            return llm_models, embedding_models
        else:
            logger.warning(f"Ollama list command failed: {result.stderr}")
            return [], []
            
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return [], []

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting RAG Application...")
    
    load_config()
    load_metadata()
    
    try:
        vector_store.load()
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load vector store: {e}")
    
    logger.info("RAG Application started successfully")

@app.get("/health", tags=["Health"])
async def health_check():
    """Check system health and configuration."""
    try:
        ollama_available, ollama_message = check_ollama_available()
        stats = vector_store.get_stats()
        
        response = {
            "status": "healthy" if ollama_available else "degraded",
            "timestamp": datetime.now().isoformat(),
            "ollama_status": {
                "available": ollama_available,
                "message": ollama_message
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
        
        logger.info(f"Health check completed: {response['status']}")
        return response
        
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

def process_document_background(filename: str, file_path: Path, file_size: int):
    """Process document in background."""
    try:
        logger.info(f"Background processing started for {filename}")
        
        if filename in document_metadata:
            document_metadata[filename]["status"] = "processing"
            save_metadata()
        
        logger.debug(f"Loading document content from {file_path}")
        document_content = load_document(file_path)
        
        logger.debug(f"Splitting document into chunks")
        documents = split_text(document_content, filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        logger.debug(f"Generating embeddings for {len(documents)} chunks")
        embeddings_model = get_embeddings_model()
        
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        logger.debug(f"Adding documents to vector store")
        vector_store.add_documents(documents, embeddings)
        
        document_metadata[filename].update({
            "chunks": len(documents),
            "status": "processed"
        })
        
        save_metadata()
        vector_store.save()
        
        logger.info(f"Background processing completed for {filename}: {len(documents)} chunks")
        
    except Exception as e:
        logger.error(f"Error in background processing for {filename}: {e}")
        logger.debug(traceback.format_exc())
        
        if filename in document_metadata:
            document_metadata[filename]["status"] = "failed"
            document_metadata[filename]["error"] = str(e)
            save_metadata()
        
        if file_path.exists():
            file_path.unlink()

@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document synchronously."""
    logger.info(f"Upload request for file: {file.filename}")
    
    if not validate_file_type(file.filename):
        logger.warning(f"Invalid file type for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file.filename in document_metadata:
        logger.warning(f"Duplicate upload attempt for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists. Please delete it first or rename your file."
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        logger.debug(f"Saving file to {file_path}")
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB"
            )
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.debug(f"Loading document content from {file_path}")
        document_content = load_document(file_path)
        
        logger.debug(f"Splitting document into chunks")
        documents = split_text(document_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        logger.debug(f"Generating embeddings for {len(documents)} chunks")
        embeddings_model = get_embeddings_model()
        
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        logger.debug(f"Adding documents to vector store")
        vector_store.add_documents(documents, embeddings)
        
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
        
        logger.info(f"Successfully processed {file.filename}: {len(documents)} chunks, {file_size} bytes")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=file_size,
            message=f"Document processed successfully into {len(documents)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Error processing upload {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        if file_path.exists():
            file_path.unlink()
        
        if file.filename in document_metadata:
            del document_metadata[file.filename]
            save_metadata()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

@app.post("/upload/async", tags=["Documents"])
async def upload_document_async(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a document and process it in the background."""
    logger.info(f"Async upload request for file: {file.filename}")
    
    if not validate_file_type(file.filename):
        logger.warning(f"Invalid file type for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file.filename in document_metadata:
        logger.warning(f"Duplicate upload attempt for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists. Please delete it first or rename your file."
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        logger.debug(f"Saving file to {file_path}")
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB"
            )
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        document_metadata[file.filename] = {
            "filename": file.filename,
            "size": file_size,
            "chunks": 0,
            "status": "pending",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        }
        save_metadata()
        
        background_tasks.add_task(process_document_background, file.filename, file_path, file_size)
        
        logger.info(f"File {file.filename} uploaded successfully, processing in background")
        
        return {
            "status": "pending",
            "filename": file.filename,
            "file_size": file_size,
            "message": f"Document uploaded successfully. Processing in background..."
        }
        
    except Exception as e:
        logger.error(f"Error during async upload {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        if file_path.exists():
            file_path.unlink()
        
        if file.filename in document_metadata:
            del document_metadata[file.filename]
            save_metadata()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )

@app.post("/query", tags=["Query"])
async def query_documents(request: QueryRequest):
    """Query the document collection."""
    logger.info(f"Query request: '{request.question[:50]}...' with model {request.model or config.model}, stream={request.stream}")
    
    start_time = datetime.now()
    
    try:
        stats = vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents available for querying. Please upload some documents first."
            )
        
        logger.debug("Generating query embedding")
        embeddings_model = get_embeddings_model()
        query_embedding = embeddings_model.embed_query(request.question)
        
        logger.debug(f"Performing similarity search with top_k={request.top_k}")
        similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
        if not similar_docs:
            raise HTTPException(
                status_code=400,
                detail="No relevant documents found for your query."
            )
        
        context_parts = []
        sources = []
        similarity_scores = []
        
        for doc, score in similar_docs:
            context_parts.append(f"Source: {doc['metadata']['source']}\nContent: {doc['page_content']}")
            sources.append(doc['metadata']['source'])
            similarity_scores.append(float(score))
        
        context = "\n\n".join(context_parts)
        
        logger.debug(f"Generating answer using LLM model {request.model or config.model}")
        llm = get_llm_model(request.model)
        
        if request.temperature is not None:
            llm.temperature = request.temperature
        
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
        
        if request.stream:
            async def generate():
                try:
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": similarity_scores,
                        "type": "metadata"
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"
                    
                    full_answer = ""
                    
                    for chunk in llm.stream(prompt):
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        full_answer += content
                    
                    cleaned_answer = clean_llm_response(full_answer)
                    
                    chunk_size = 50
                    for i in range(0, len(cleaned_answer), chunk_size):
                        chunk_text = cleaned_answer[i:i + chunk_size]
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk_text})}\n\n"
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    completion = {
                        "type": "done",
                        "processing_time": processing_time
                    }
                    yield f"data: {json.dumps(completion)}\n\n"
                    
                    config.total_queries += 1
                    save_config()  # ADDED: Save config after updating query count
                    logger.info(f"Streaming query completed in {processing_time:.2f}s, {len(similar_docs)} chunks used")
                except Exception as e:
                    logger.error(f"# Streaming error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    error_msg = {
                        "type": "error",
                        "error": str(e)
                    }
                    yield f"data: {json.dumps(error_msg)}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            answer = clean_llm_response(answer)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            config.total_queries += 1
            save_config()  # ADDED: Save config after updating query count
            
            logger.info(f"Query completed in {processing_time:.2f}s, {len(similar_docs)} chunks used")
            
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
    
    documents = []
    for filename, metadata in document_metadata.items():
        documents.append(DocumentInfo(**metadata))
    
    logger.info(f"Listed {len(documents)} documents")
    return documents

@app.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a specific document."""
    logger.info(f"Delete request for document: {filename}")
    
    if filename not in document_metadata:
        logger.warning(f"Document not found: {filename}")
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found"
        )
    
    try:
        vector_store.remove_documents_by_source(filename)
        
        del document_metadata[filename]
        save_metadata()
        
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
        
        global document_metadata
        document_metadata = {}
        save_metadata()
        
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
    
    if filename not in document_metadata:
        logger.warning(f"Document not found for preview: {filename}")
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
        
        chunks_data = []
        for doc in preview_chunks:
            chunks_data.append({
                "chunk_id": doc['metadata']['chunk_id'],
                "content": doc['page_content'][:500] + ("..." if len(doc['page_content']) > 500 else ""),
                "length": len(doc['page_content'])
            })
        
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
    
    changed_fields = []
    
    try:
        if config_update.model is not None:
            if config.model != config_update.model:
                config.model = config_update.model
                changed_fields.append("model")
                global llm_model, llm_model_name
                llm_model = None
                llm_model_name = None
        
        if config_update.embedding_model is not None:
            if config.embedding_model != config_update.embedding_model:
                config.embedding_model = config_update.embedding_model
                changed_fields.append("embedding_model")
                global embeddings_model
                embeddings_model = None
        
        if config_update.chunk_size is not None:
            if config.chunk_size != config_update.chunk_size:
                config.chunk_size = config_update.chunk_size
                changed_fields.append("chunk_size")
        
        if config_update.chunk_overlap is not None:
            if config.chunk_overlap != config_update.chunk_overlap:
                config.chunk_overlap = config_update.chunk_overlap
                changed_fields.append("chunk_overlap")
        
        if config_update.temperature is not None:
            if config.temperature != config_update.temperature:
                config.temperature = config_update.temperature
                changed_fields.append("temperature")
                if llm_model:
                    llm_model.temperature = config.temperature
        
        save_config()
        
        logger.info(f"Configuration updated successfully, changed fields: {changed_fields}")
        
        response = {
            "message": "Configuration updated successfully",
            "changed_fields": changed_fields
        }
        
        if "embedding_model" in changed_fields:
            response["warning"] = "Embedding model changed. Consider rebuilding vectors for existing documents."
        
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
        
        if not llm_models:
            llm_models = ["phi3", "llama3", "mistral"]
        if not embedding_models:
            embedding_models = ["nomic-embed-text"]
        
        response = {
            "llm_models": llm_models,
            "embedding_models": embedding_models,
            "current_llm": config.model,
            "current_embedding": config.embedding_model
        }
        
        logger.debug(f"Models response: {len(llm_models)} LLM, {len(embedding_models)} embedding models")
        return response
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {
            "llm_models": ["phi3", "llama3", "mistral"],
            "embedding_models": ["nomic-embed-text"],
            "current_llm": config.model,
            "current_embedding": config.embedding_model
        }

@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get system statistics."""
    logger.debug("Statistics request")
    
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
        logger.error(f"Error getting statistics: {e}")
        return {
            "total_documents": len(document_metadata),
            "total_chunks": 0,
            "total_queries": config.total_queries,
            "total_storage_size": 0,
            "average_chunks_per_document": 0.0,
            "last_update": None
        }

@app.get("/debug/embeddings", tags=["Debug"])
async def debug_embeddings(text: str = Query("This is a test sentence")):
    """Test embedding generation."""
    logger.debug(f"Debug embeddings request with text: '{text[:50]}...'")
    
    try:
        start_time = datetime.now()
        embeddings_model = get_embeddings_model()
        embedding = embeddings_model.embed_query(text)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "model": config.embedding_model,
            "dimensions": len(embedding),
            "generation_time": generation_time,
            "embedding": embedding[:10] if embedding else []
        }
        
    except Exception as e:
        logger.error(f"Error in debug embeddings: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {str(e)}"
        )

@app.get("/debug/vector-store", tags=["Debug"])
async def debug_vector_store():
    """Inspect vector store state."""
    logger.debug("Debug vector store request")
    
    try:
        stats = vector_store.get_stats()
        sample_docs = vector_store.get_sample_documents(5)
        
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
        logger.error(f"Error in debug vector store: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to inspect vector store: {str(e)}"
        )

@app.post("/rebuild-vectors", tags=["Debug"])
async def rebuild_vectors():
    """Rebuild all vectors with current embedding model."""
    logger.info("Rebuild vectors request")
    
    try:
        vector_store.clear()
        
        results = {}
        
        for filename in list(document_metadata.keys()):
            try:
                logger.debug(f"Rebuilding vectors for {filename}")
                
                file_path = UPLOAD_DIR / filename
                if not file_path.exists():
                    results[filename] = {"success": False, "error": "File not found"}
                    continue
                
                document_content = load_document(file_path)
                documents = split_text(document_content, filename)
                
                embeddings_model = get_embeddings_model()
                texts = [doc["page_content"] for doc in documents]
                embeddings = embeddings_model.embed_documents(texts)
                
                vector_store.add_documents(documents, embeddings)
                
                document_metadata[filename]["chunks"] = len(documents)
                
                results[filename] = {"success": True, "chunks": len(documents)}
                logger.debug(f"Successfully rebuilt {filename}: {len(documents)} chunks")
                
            except Exception as e:
                logger.error(f"Error rebuilding {filename}: {e}")
                results[filename] = {"success": False, "error": str(e)}
        
        save_metadata()
        vector_store.save()
        
        success_count = sum(1 for result in results.values() if result["success"])
        
        logger.info(f"Rebuild completed: {success_count}/{len(results)} documents successful")
        
        return {
            "message": f"Rebuild completed: {success_count}/{len(results)} documents processed successfully",
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
        "rag_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
