"""
RAG Backend - Main Application Entry Point
Modular RAG system with FastAPI, Ollama, and LangChain
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vector_store import VectorStore
from backend.config import ConfigManager, UPLOAD_DIR, VECTOR_DIR
from backend.managers import MetadataManager, ModelManager
from backend.document_processor import DocumentProcessor
from backend.routes import router, init_managers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_app.log')
    ]
)
logger = logging.getLogger(__name__)

config_manager = ConfigManager()
metadata_manager = MetadataManager()
vector_store = VectorStore()
model_manager = ModelManager(config_manager)
document_processor = DocumentProcessor(config_manager)

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting RAG Application...")
    
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
    description="Modular RAG system with Ollama and LangChain",
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

init_managers(config_manager, metadata_manager, model_manager, document_processor, vector_store)

app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting RAG Backend Server on port 8000...")
    uvicorn.run(
        "rag_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
