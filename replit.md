# Overview

A production-ready Retrieval-Augmented Generation (RAG) system that enables users to upload documents (PDF, TXT, DOCX), convert them into vector embeddings using local Ollama models, and perform intelligent question-answering. The system consists of a FastAPI backend for document processing and vector storage, and a Streamlit frontend for user interaction.

**Key Capabilities:**
- Document upload and processing with automatic text chunking
- Local LLM integration via Ollama (no API keys required)
- Custom in-memory vector database with JSON persistence
- Real-time streaming responses
- Multi-document chat with conversation history
- Document management and export capabilities

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Architecture (FastAPI)

**Core Framework:** FastAPI with async/await support for concurrent document processing

**Modular Structure:**
- `rag_backend.py` - Main application entry point with lifespan management
- `backend/routes.py` - API endpoint handlers
- `backend/config.py` - Centralized configuration management with JSON persistence
- `backend/managers.py` - Metadata and model lifecycle management
- `backend/document_processor.py` - Document loading and text splitting
- `backend/models.py` - Pydantic models and custom Ollama LLM client
- `backend/utils.py` - Utility functions for validation and health checks
- `vector_store.py` - Custom vector database implementation

**Document Processing Pipeline:**
1. File validation (extension, size limits)
2. Document loading via LangChain loaders (PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader)
3. Text splitting using RecursiveCharacterTextSplitter (configurable chunk size/overlap)
4. Embedding generation via Ollama
5. Vector storage with metadata persistence

**Custom Ollama LLM Client:**
- Auto-detects API endpoint type (`/api/chat` vs `/api/generate`)
- Supports both streaming and non-streaming responses
- Handles model warm-up and cold-start timeouts
- Graceful fallback between endpoints
- Cleans reasoning tags from model outputs

**Vector Store Design:**
- In-memory numpy arrays for fast similarity search
- Cosine similarity using scikit-learn
- JSON-based persistence (documents + embeddings)
- Validates embedding dimensions on insertion
- Tracks last update timestamp

**API Endpoints:**
- `GET /health` - System health and configuration status
- `POST /upload` - Document upload with background processing
- `POST /query` - Question answering (streaming/non-streaming)
- `GET /documents` - List uploaded documents with metadata
- `DELETE /documents/{filename}` - Remove document and vectors
- `GET /models` - Available Ollama models
- `POST /config` - Update system configuration

## Frontend Architecture (Streamlit)

**Modular Structure:**
- `frontend/app.py` - Main entry point and layout orchestration
- `frontend/api_client.py` - Centralized backend API client with error handling
- `frontend/session_state.py` - Session state management and conversation history
- `frontend/chat.py` - Chat interface with export and citations
- `frontend/sidebar.py` - Document management UI
- `frontend/onboarding.py` - First-time user experience
- `frontend/toast.py` - Queue-based notification system
- `frontend/styles.py` - Custom CSS styling
- `frontend/config.py` - Frontend configuration

**Session State Management:**
- Per-document chat histories (`document_chats` dictionary)
- Conversation persistence and export (JSON/Markdown)
- Suggested questions per document
- Upload state tracking with unique keys

**UI Components:**
- Document cards with selection and deletion
- Streaming chat interface with thinking indicators
- Citation display showing source chunks
- Export options (JSON/Markdown)
- Suggested questions based on context
- Responsive design with custom CSS

**API Client Pattern:**
- Unified request handler with timeout management
- Session-based connections for efficiency
- Structured error responses
- Streaming support for long-running queries

## Configuration Management

**Centralized Config (ConfigManager):**
- Default settings: model names, chunk parameters, temperature, timeouts
- JSON persistence in `vector_data/config.json`
- Query statistics tracking
- Thread-safe updates

**Default Configuration:**
```python
{
    'model': 'phi3',                    # Default LLM
    'embedding_model': 'nomic-embed-text',  # Default embeddings
    'chunk_size': 1000,                 # Text chunk size
    'chunk_overlap': 200,               # Overlap between chunks
    'temperature': 0.7,                 # LLM temperature
    'timeout': 120,                     # Request timeout
    'cold_start_timeout': 600           # First model load timeout
}
```

## Data Flow

**Upload Flow:**
```
User uploads file → Validation → LangChain loader → Text splitting → 
Ollama embeddings → Vector store → Metadata persistence
```

**Query Flow:**
```
User question → Ollama embeddings → Similarity search (top-k) → 
Context assembly → Ollama LLM → Stream response → Citation extraction
```

**Persistence Strategy:**
- Document metadata: `vector_data/metadata.json`
- Embeddings: `vector_data/vectors.json`
- Configuration: `vector_data/config.json`
- Uploaded files: `uploaded_documents/` directory

# External Dependencies

## LLM Infrastructure

**Ollama (Required):**
- Local LLM server running at `http://localhost:11434`
- Provides both embedding models (nomic-embed-text) and chat models (phi3, llama3, mistral, deepseek-r1)
- Automatic endpoint detection (/api/chat vs /api/generate)
- Model management via `/api/tags` endpoint

## LangChain Framework

**Core Components:**
- `langchain` - Main framework for RAG pipeline orchestration
- `langchain-community` - Document loaders (PDF, TXT, DOCX)
- `langchain-ollama` - Ollama integration for embeddings and chat
- `langchain-text-splitters` - RecursiveCharacterTextSplitter for chunking

**Design Decision:** LangChain provides standardized interfaces for document loading and text splitting, making it easy to swap implementations or add new document types.

## Document Processing

**Libraries:**
- `pypdf` - PDF parsing and text extraction
- `python-docx` + `docx2txt` - DOCX document processing
- `unstructured` - Additional document format support (fallback)

**File Type Support:**
- `.pdf` → PyPDFLoader
- `.txt` → TextLoader
- `.docx` → UnstructuredWordDocumentLoader

## Vector Similarity

**scikit-learn:**
- Cosine similarity computation for vector search
- Chosen for simplicity and no additional database dependencies
- In-memory approach suitable for moderate document collections

**Alternative Considered:** Vector databases (Pinecone, Weaviate, ChromaDB) were considered but rejected to keep the system self-contained and dependency-free.

## Web Frameworks

**FastAPI:**
- Modern async framework for backend API
- Automatic OpenAPI documentation
- Pydantic integration for request/response validation
- Background task support for async document processing

**Streamlit:**
- Rapid UI development for chat interface
- Built-in session state management
- Easy integration with Python backend

**Uvicorn:**
- ASGI server for FastAPI
- Runs backend on `http://localhost:8000`

## Data Processing

**NumPy:**
- Array operations for embeddings
- Efficient storage and manipulation of vectors

**Requests:**
- HTTP client for Ollama API communication
- Session-based connection pooling
- Timeout management

## No Database Requirement

**Design Decision:** The system uses JSON file persistence instead of a traditional database (PostgreSQL, SQLite) to maintain simplicity and portability. This approach is suitable for moderate document collections but may require migration to a proper vector database for production scale.

**Trade-offs:**
- **Pros:** Zero database setup, portable, simple deployment
- **Cons:** Limited scalability, no concurrent write safety, full reload on restart
- **Future Migration Path:** Easy to swap vector_store.py implementation with Pinecone, Weaviate, or PostgreSQL with pgvector extension