import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
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
    source_filter: Optional[str] = Field(None, description="Filter by document source")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    use_hybrid_search: bool = Field(False, description="Use hybrid semantic + keyword search")
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

config = Config()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Assistant API",
    description="Production-ready Retrieval-Augmented Generation API with document upload, embedding generation, and intelligent querying using local Ollama LLMs",
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
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}

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
    """Get or create embeddings model."""
    global embeddings_model
    try:
        if embeddings_model is None:
            logger.info(f"Initializing embeddings model: {config.embedding_model}")
            embeddings_model = OllamaEmbeddings(model=config.embedding_model)
            
            # Test the model
            test_embedding = embeddings_model.embed_query("test")
            logger.info(f"Embeddings model initialized successfully, dimensions: {len(test_embedding)}")
        
        return embeddings_model
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize embeddings model '{config.embedding_model}'. Please ensure Ollama is running and the model is available."
        )

def get_llm_model(model_name: str = None):
    """Get or create LLM model."""
    global llm_model
    try:
        model_to_use = model_name or config.model
        
        if llm_model is None or llm_model.model != model_to_use:
            logger.info(f"Initializing LLM model: {model_to_use}")
            llm_model = ChatOllama(
                model=model_to_use,
                temperature=config.temperature
            )
        
        return llm_model
    except Exception as e:
        logger.error(f"Error initializing LLM model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LLM model '{model_to_use}'. Please ensure Ollama is running and the model is available."
        )

def validate_file_type(filename: str) -> bool:
    """Validate file type based on extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def load_document(file_path: Path) -> List[str]:
    """Load document content based on file type."""
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
    """Split text content into chunks."""
    try:
        # Combine all pages
        full_text = "\n\n".join(content)
        
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
        
        logger.info(f"Split {filename} into {len(documents)} chunks")
        return documents
        
    except Exception as e:
        logger.error(f"Error splitting text for {filename}: {e}")
        raise

def check_ollama_available():
    """Check if Ollama is available and models are accessible."""
    try:
        # Try to initialize embeddings model
        embeddings = OllamaEmbeddings(model=config.embedding_model)
        test_embedding = embeddings.embed_query("test")
        
        # Try to initialize LLM
        llm = ChatOllama(model=config.model)
        
        return True, f"Ollama available with {config.embedding_model} and {config.model}"
    except Exception as e:
        return False, str(e)

def get_available_models():
    """Get available models from Ollama."""
    try:
        import subprocess
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
            
            return llm_models, embedding_models
        else:
            logger.warning(f"Ollama list command failed: {result.stderr}")
            return [], []
            
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return [], []

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting RAG Application...")
    
    # Load existing data
    load_metadata()
    
    try:
        vector_store.load()
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load vector store: {e}")
    
    logger.info("RAG Application started successfully")

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Check system health and configuration."""
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

# Background processing function
async def process_document_background(filename: str, file_path: Path, file_size: int):
    """Process document in background."""
    try:
        logger.info(f"Background processing started for {filename}")
        
        # Update status to processing
        if filename in document_metadata:
            document_metadata[filename]["status"] = "processing"
            save_metadata()
        
        # Load and process document
        logger.debug(f"Loading document content from {file_path}")
        document_content = load_document(file_path)
        
        # Split into chunks
        logger.debug(f"Splitting document into chunks")
        documents = split_text(document_content, filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        logger.debug(f"Generating embeddings for {len(documents)} chunks")
        embeddings_model = get_embeddings_model()
        
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        # Add to vector store
        logger.debug(f"Adding documents to vector store")
        vector_store.add_documents(documents, embeddings)
        
        # Update metadata
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
        
        # Update status to failed
        if filename in document_metadata:
            document_metadata[filename]["status"] = "failed"
            document_metadata[filename]["error"] = str(e)
            save_metadata()
        
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()

# Upload endpoint (sync)
@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document synchronously."""
    logger.info(f"Upload request for file: {file.filename}")
    
    # Validate file type
    if not validate_file_type(file.filename):
        logger.warning(f"Invalid file type for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check for duplicates
    if file.filename in document_metadata:
        logger.warning(f"Duplicate upload attempt for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists. Please delete it first or rename your file."
        )
    
    file_path = UPLOAD_DIR / file.filename
    temp_path = None
    
    try:
        # Save uploaded file
        logger.debug(f"Saving file to {file_path}")
        content = await file.read()
        file_size = len(content)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Load and process document
        logger.debug(f"Loading document content from {file_path}")
        document_content = load_document(file_path)
        
        # Split into chunks
        logger.debug(f"Splitting document into chunks")
        documents = split_text(document_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        logger.debug(f"Generating embeddings for {len(documents)} chunks")
        embeddings_model = get_embeddings_model()
        
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        # Add to vector store
        logger.debug(f"Adding documents to vector store")
        vector_store.add_documents(documents, embeddings)
        
        # Save metadata
        document_metadata[file.filename] = {
            "filename": file.filename,
            "size": file_size,
            "chunks": len(documents),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()  # Remove dot from extension
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
        
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()
        if temp_path and temp_path.exists():
            temp_path.unlink()
        
        # Remove from metadata if added
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
    logger.info(f"Query request: '{request.question[:50]}...' with model {request.model}, stream={request.stream}")
    
    start_time = datetime.now()
    
    try:
        # Check if vector store has documents
        stats = vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents available for querying. Please upload some documents first."
            )
        
        # Generate query embedding
        logger.debug("Generating query embedding")
        embeddings_model = get_embeddings_model()
        query_embedding = embeddings_model.embed_query(request.question)
        
        # Perform search (hybrid or semantic)
        if request.use_hybrid_search:
            logger.debug(f"Performing hybrid search with top_k={request.top_k}")
            similar_docs = vector_store.hybrid_search(
                query_text=request.question,
                query_embedding=query_embedding,
                k=request.top_k,
                source_filter=request.source_filter,
                similarity_threshold=request.similarity_threshold
            )
        else:
            logger.debug(f"Performing similarity search with top_k={request.top_k}")
            similar_docs = vector_store.similarity_search(
                query_embedding, 
                k=request.top_k,
                source_filter=request.source_filter,
                similarity_threshold=request.similarity_threshold
            )
        
        if not similar_docs:
            raise HTTPException(
                status_code=400,
                detail="No relevant documents found for your query."
            )
        
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
        logger.debug(f"Generating answer using LLM model {request.model or config.model}")
        llm = get_llm_model(request.model)
        
        if request.temperature is not None:
            llm.temperature = request.temperature
        
        prompt = f"""Based on the following context from uploaded documents, answer the question. If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {request.question}

Answer:"""
        
        # Stream or regular response
        if request.stream:
            # Streaming response
            async def generate():
                # Send metadata first
                metadata = {
                    "sources": sources,
                    "chunks_used": len(similar_docs),
                    "similarity_scores": similarity_scores,
                    "type": "metadata"
                }
                yield f"data: {json.dumps(metadata)}\n\n"
                
                # Stream the answer
                full_answer = ""
                for chunk in llm.stream(prompt):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_answer += content
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                
                # Send completion
                processing_time = (datetime.now() - start_time).total_seconds()
                completion = {
                    "type": "done",
                    "processing_time": processing_time
                }
                yield f"data: {json.dumps(completion)}\n\n"
                
                # Update query count
                config.total_queries += 1
                logger.info(f"Streaming query completed in {processing_time:.2f}s, {len(similar_docs)} chunks used")
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Regular response
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update query count
            config.total_queries += 1
            
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

# List documents endpoint
@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all uploaded documents."""
    logger.debug("Listing documents")
    
    documents = []
    for filename, metadata in document_metadata.items():
        documents.append(DocumentInfo(**metadata))
    
    logger.info(f"Listed {len(documents)} documents")
    return documents

# Delete document endpoint
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
        # Remove from vector store
        vector_store.remove_documents_by_source(filename)
        
        # Remove metadata
        del document_metadata[filename]
        save_metadata()
        
        # Remove file
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        # Save vector store
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

# Clear all documents endpoint
@app.delete("/clear", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents and embeddings."""
    logger.info("Clear all documents request")
    
    try:
        # Clear vector store
        vector_store.clear()
        
        # Clear metadata
        global document_metadata
        document_metadata = {}
        save_metadata()
        
        # Remove all uploaded files
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        # Save empty vector store
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

# Preview document endpoint
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

# Configuration endpoint
@app.post("/configure", tags=["Configuration"])
async def configure_system(config_update: ModelConfig):
    """Update system configuration."""
    logger.info(f"Configuration update request: {config_update.dict(exclude_none=True)}")
    
    changed_fields = []
    
    try:
        # Update configuration
        if config_update.model is not None:
            if config.model != config_update.model:
                config.model = config_update.model
                changed_fields.append("model")
                global llm_model
                llm_model = None  # Reset to force reinitialization
        
        if config_update.embedding_model is not None:
            if config.embedding_model != config_update.embedding_model:
                config.embedding_model = config_update.embedding_model
                changed_fields.append("embedding_model")
                global embeddings_model
                embeddings_model = None  # Reset to force reinitialization
        
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

# Get models endpoint
@app.get("/models", tags=["Configuration"])
async def get_models():
    """Get available models and current configuration."""
    logger.debug("Get models request")
    
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
        
        logger.debug(f"Models response: {len(llm_models)} LLM, {len(embedding_models)} embedding models")
        return response
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
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

# Debug embedding endpoint
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
            "embedding": embedding[:10] if embedding else []  # First 10 values for preview
        }
        
    except Exception as e:
        logger.error(f"Error in debug embeddings: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {str(e)}"
        )

# Debug vector store endpoint
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

# Rebuild vectors endpoint
@app.post("/rebuild-vectors", tags=["Debug"])
async def rebuild_vectors():
    """Rebuild all vectors with current embedding model."""
    logger.info("Rebuild vectors request")
    
    try:
        # Clear existing vectors
        vector_store.clear()
        
        results = {}
        
        for filename in list(document_metadata.keys()):
            try:
                logger.debug(f"Rebuilding vectors for {filename}")
                
                # Load document
                file_path = UPLOAD_DIR / filename
                if not file_path.exists():
                    results[filename] = {"success": False, "error": "File not found"}
                    continue
                
                document_content = load_document(file_path)
                documents = split_text(document_content, filename)
                
                # Generate embeddings
                embeddings_model = get_embeddings_model()
                texts = [doc["page_content"] for doc in documents]
                embeddings = embeddings_model.embed_documents(texts)
                
                # Add to vector store
                vector_store.add_documents(documents, embeddings)
                
                # Update metadata
                document_metadata[filename]["chunks"] = len(documents)
                
                results[filename] = {"success": True, "chunks": len(documents)}
                logger.debug(f"Successfully rebuilt {filename}: {len(documents)} chunks")
                
            except Exception as e:
                logger.error(f"Error rebuilding {filename}: {e}")
                results[filename] = {"success": False, "error": str(e)}
        
        # Save everything
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
