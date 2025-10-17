# Dependency Documentation

This document provides detailed information about all dependencies used in the RAG Document Assistant application, including their purpose and inner dependencies.

## Direct Dependencies (from pyproject.toml)

### 1. FastAPI (>=0.119.0)
**Purpose**: Web framework for building the backend API

**Inner Dependencies**:
- `starlette` - ASGI framework that FastAPI is built on
- `pydantic` - Data validation (also a direct dependency)
- `typing-extensions` - Backported type hints

**Used For**:
- REST API endpoints (`/health`, `/upload`, `/query`, etc.)
- Request/response handling
- Dependency injection
- CORS middleware
- Background tasks for async document processing

**Files**: `rag_backend.py`

---

### 2. Uvicorn (>=0.37.0)
**Purpose**: ASGI server to run FastAPI application

**Inner Dependencies**:
- `click` - Command line interface
- `h11` - HTTP/1.1 protocol implementation
- `uvloop` (optional) - Fast event loop

**Used For**:
- Running the FastAPI backend server on port 8000
- Handling HTTP requests and WebSocket connections

**Files**: `rag_backend.py` (startup)

---

### 3. Streamlit (>=1.50.0)
**Purpose**: Interactive web UI framework for the frontend

**Inner Dependencies**:
- `altair` - Declarative visualization
- `pandas` - Data manipulation
- `pillow` - Image processing
- `tornado` - Web server
- `watchdog` - File system monitoring
- `click` - CLI interface
- `protobuf` - Data serialization

**Used For**:
- Building the user interface
- Document upload interface
- Query interface
- Dashboard and configuration pages
- Session state management
- Responsive layout and styling

**Files**: `app.py`

---

### 4. LangChain (>=0.3.27)
**Purpose**: Framework for building LLM applications

**Inner Dependencies**:
- `pydantic` - Data validation
- `sqlalchemy` - Database toolkit
- `aiohttp` - Async HTTP client
- `dataclasses-json` - JSON serialization
- `tenacity` - Retry library

**Used For**:
- Text splitting with `RecursiveCharacterTextSplitter`
- Document loading abstractions
- LLM interaction patterns

**Files**: `rag_backend.py`

---

### 5. LangChain Community (>=0.3.31)
**Purpose**: Community-maintained LangChain integrations

**Inner Dependencies**:
- All LangChain dependencies
- Various document loader dependencies

**Used For**:
- `PyPDFLoader` - Loading PDF documents
- `TextLoader` - Loading text files
- `UnstructuredWordDocumentLoader` - Loading DOCX files

**Files**: `rag_backend.py`

---

### 6. LangChain Ollama (>=0.3.10)
**Purpose**: Ollama integration for LangChain

**Inner Dependencies**:
- `langchain-core` - Core LangChain functionality
- `ollama` - Ollama Python client (also direct dependency)

**Used For**:
- `OllamaEmbeddings` - Generating text embeddings
- `ChatOllama` - LLM chat completions with streaming

**Files**: `rag_backend.py`

---

### 7. Ollama (>=0.6.0)
**Purpose**: Python client for Ollama API

**Inner Dependencies**:
- `httpx` - Modern HTTP client
- `pydantic` - Data validation

**Used For**:
- Communicating with local Ollama server
- Pulling and managing models
- Generating embeddings and completions

**Files**: `rag_backend.py` (via langchain-ollama)

---

### 8. NumPy (>=2.3.4)
**Purpose**: Numerical computing library

**Inner Dependencies**:
- Native C/Fortran libraries for performance

**Used For**:
- Vector/array operations for embeddings
- Cosine similarity calculations
- Numerical computations in vector store

**Files**: `rag_backend.py`, `vector_store.py`

---

### 9. Scikit-learn (>=1.7.2)
**Purpose**: Machine learning library

**Inner Dependencies**:
- `numpy` - Numerical operations (also direct dependency)
- `scipy` - Scientific computing
- `joblib` - Parallel processing
- `threadpoolctl` - Thread control

**Used For**:
- `cosine_similarity` - Calculating similarity between embeddings
- Vector similarity search in the custom vector store

**Files**: `rag_backend.py`, `vector_store.py`

---

### 10. PyPDF (>=6.1.1)
**Purpose**: PDF document parsing

**Inner Dependencies**:
- `typing-extensions` - Type hints

**Used For**:
- Extracting text content from PDF files
- PDF metadata extraction

**Files**: `rag_backend.py` (via LangChain loaders)

---

### 11. Python-DOCX (>=1.2.0)
**Purpose**: DOCX document parsing

**Inner Dependencies**:
- `lxml` - XML processing
- `Pillow` - Image handling

**Used For**:
- Reading Microsoft Word documents
- Extracting text from DOCX files

**Files**: `rag_backend.py` (via LangChain loaders)

---

### 12. Docx2txt (>=0.9)
**Purpose**: Additional DOCX text extraction

**Inner Dependencies**:
- Minimal dependencies

**Used For**:
- Fallback DOCX text extraction
- Handling complex DOCX structures

**Files**: `rag_backend.py` (via LangChain loaders)

---

### 13. Requests (>=2.32.5)
**Purpose**: HTTP library for making API calls

**Inner Dependencies**:
- `urllib3` - HTTP client
- `charset-normalizer` - Character encoding detection
- `certifi` - SSL certificates
- `idna` - Internationalized domain names

**Used For**:
- Frontend to backend API communication
- HTTP POST/GET/DELETE requests
- Streaming response handling

**Files**: `app.py`

---

### 14. Python-Multipart (>=0.0.20)
**Purpose**: File upload handling

**Inner Dependencies**:
- Minimal dependencies

**Used For**:
- Parsing multipart/form-data requests
- Handling file uploads in FastAPI

**Files**: `rag_backend.py`

---

### 15. Pydantic (>=2.12.2)
**Purpose**: Data validation and settings management

**Inner Dependencies**:
- `typing-extensions` - Type hints
- `pydantic-core` - Core validation logic (Rust-based)

**Used For**:
- Request/response models in FastAPI
- Data validation and serialization
- Type checking and error handling

**Files**: `rag_backend.py`

---

## Python Standard Library Modules

These are built-in Python modules that don't require installation:

### Core Modules Used:
- **json** - JSON encoding/decoding for data persistence
- **os** - Operating system interfaces, environment variables, file paths
- **time** - Time-related functions, delays
- **logging** - Application logging and debugging
- **datetime** - Date and time operations, timestamps
- **pathlib** - Object-oriented filesystem paths
- **typing** - Type hints (List, Dict, Any, Optional, Tuple)
- **traceback** - Exception stack traces for error handling

---

## Dependency Graph Summary

```
Application Layer
├── app.py (Streamlit Frontend)
│   ├── streamlit
│   ├── requests
│   └── Python stdlib (json, time, os, typing)
│
└── rag_backend.py (FastAPI Backend)
    ├── fastapi
    │   └── starlette, pydantic
    ├── uvicorn
    ├── langchain
    │   └── pydantic, sqlalchemy, aiohttp
    ├── langchain-community
    │   ├── pypdf
    │   ├── python-docx (→ lxml, pillow)
    │   └── docx2txt
    ├── langchain-ollama
    │   └── ollama (→ httpx)
    ├── numpy
    ├── scikit-learn (→ scipy, joblib)
    └── Python stdlib (os, json, logging, datetime, pathlib, typing, traceback)

Data Layer
└── vector_store.py (Custom Vector Store)
    ├── numpy
    ├── scikit-learn
    └── Python stdlib (json, logging, pathlib, typing, datetime)
```

---

## Installation Commands

### Using pip (recommended):
```bash
pip install fastapi>=0.119.0 uvicorn>=0.37.0 streamlit>=1.50.0 \
  langchain>=0.3.27 langchain-community>=0.3.31 langchain-ollama>=0.3.10 \
  ollama>=0.6.0 numpy>=2.3.4 scikit-learn>=1.7.2 \
  pypdf>=6.1.1 python-docx>=1.2.0 docx2txt>=0.9 \
  requests>=2.32.5 python-multipart>=0.0.20 pydantic>=2.12.2
```

### Using pyproject.toml:
```bash
pip install -e .
```

---

## External Service Dependencies

### Ollama (Required)
- **Purpose**: Local LLM and embedding model server
- **Installation**: Download from https://ollama.ai
- **Required Models**:
  - `nomic-embed-text` - Text embedding model
  - `llama3` - Default language model
  - `mistral` - Alternative language model
  - `phi` - Lightweight language model

**Setup Commands**:
```bash
# Install Ollama first, then pull models:
ollama pull nomic-embed-text
ollama pull llama3
ollama pull mistral
ollama pull phi
```

---

## Dependency Size Estimates

Approximate download/installation sizes:

- **Small** (<10 MB): python-multipart, docx2txt, pydantic, requests
- **Medium** (10-50 MB): fastapi, uvicorn, pypdf, python-docx, ollama
- **Large** (50-200 MB): numpy, scikit-learn, langchain packages
- **Very Large** (>200 MB): streamlit (with all visualization dependencies)

**Total estimated size**: ~800 MB - 1.2 GB (including all dependencies)

---

## Version Compatibility Notes

1. **Python 3.11+** required for optimal performance
2. **NumPy 2.x** may have breaking changes from 1.x
3. **Pydantic 2.x** has significant API changes from 1.x
4. **LangChain** versions should stay synchronized (core, community, ollama)
5. **Streamlit** 1.50+ includes improved mobile responsiveness

---

## Security Considerations

1. **No API Keys Required**: Using local Ollama means no external API dependencies
2. **File Upload Validation**: Supported formats limited to PDF, TXT, DOCX
3. **CORS Configured**: Backend allows cross-origin requests for frontend communication
4. **Local Data Storage**: All embeddings and documents stored locally

---

## Performance Notes

1. **NumPy + Scikit-learn**: Optimized C/Fortran libraries for fast vector operations
2. **Uvicorn with uvloop**: High-performance async server
3. **Streaming Responses**: Reduces perceived latency for LLM outputs
4. **Background Tasks**: Async document processing doesn't block API
5. **In-Memory Vector Store**: Fast similarity search, JSON persistence for durability
