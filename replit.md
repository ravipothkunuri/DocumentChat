# LangChain Ollama RAG Assistant

## Overview
A Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama LLMs, and Streamlit. This application allows users to upload PDF documents, generate embeddings, and query their document knowledge base with AI-powered responses.

**Current State**: Fully configured and running on Replit with optimized dependencies.

## Recent Changes (October 28, 2025)

### New Features
- **Chat Export Functionality**: Added ability to export conversations
  - Export dropdown in chat header with JSON and Markdown format options
  - Immediate download button appears beside format selector
  - Filenames include document name and timestamp for easy organization
  - JSON export includes metadata (document, timestamp, message count)
  - Markdown export creates readable formatted conversation history

### CSS Cleanup & UI Optimization
- **Minimized Custom CSS**: Removed 90% of custom styling in favor of Streamlit built-in components
  - Removed: Complex gradients, custom animations (pulse, bounce, slideIn)
  - Removed: Elaborate chat bubble styling with speech bubble tails
  - Removed: Sidebar sizing and transition animations
  - Removed: Custom scrollbar styling and loading animations
  - Kept: Essential chat alignment, delete button styling, stop button styling
- **Header Simplification**: Replaced custom HTML header with native `st.title()`
- **Delete Button Fix**: Adjusted column ratio from [5,1] to [6,1] for better alignment

### Dependency Optimizations
- **Removed**: `unstructured` (large, unused dependency)
- **Removed**: `scikit-learn` (replaced with numpy for cosine similarity)
- **Removed**: TXT and DOCX support (simplified to PDF-only)
- **Upgraded**: `pypdf` → `pymupdf` (10x faster, better quality)
- **Using**: `langchain-ollama` (recommended over deprecated `langchain-community` Ollama)

### Configuration Updates
- Updated Streamlit config to work with Replit's proxy system
- Created unified startup script (`start_services.sh`) that runs both backend and frontend
- Backend runs on `localhost:8000`, frontend on `0.0.0.0:5000`
- Deployment configured for Replit autoscale

## Project Architecture

### Backend (FastAPI)
- **Location**: `backend/`
- **Main file**: `backend/main.py`
- **Port**: 8000 (localhost only - this is correct for Replit architecture)
- **Note**: Backend binds to localhost because it only needs to communicate with the frontend Streamlit server, which runs on the same machine. User browsers connect to the frontend via Replit's proxy.
- **Key Components**:
  - `model_manager.py` - Manages LLM and embedding models
  - `document_processor.py` - Handles PDF processing with PyMuPDF
  - `routes.py` - API endpoints
  - `ollama.py` - Custom async Ollama client
  - `config_manager.py` - Configuration management

### Frontend (Streamlit)
- **Location**: `frontend/`
- **Main file**: `frontend/app.py`
- **Port**: 5000 (binds to 0.0.0.0 for Replit)
- **Key Components**:
  - `chat.py` - Chat interface with export functionality
  - `sidebar.py` - Document management UI
  - `api_client.py` - Backend API client with streaming
  - `styles.py` - Minimal custom CSS styling
  - `export_utils.py` - Conversation export utilities (JSON/Markdown)

### Vector Store
- **File**: `vector_store.py`
- **Type**: Custom in-memory vector database with JSON persistence
- **Similarity**: Numpy-based cosine similarity (optimized, no scikit-learn)
- **Storage**: `vector_data/vectors.json`

## Running the Application

### Development
The application starts automatically via the Frontend workflow which runs `start_services.sh`.

Manual start:
```bash
bash start_services.sh
```

This script:
1. Starts FastAPI backend on localhost:8000
2. Waits 3 seconds for backend initialization
3. Starts Streamlit frontend on 0.0.0.0:5000

### Deployment
Configured for Replit Autoscale deployment. Simply click "Deploy" in Replit.

## Dependencies

### Core Framework
- `fastapi` - Modern web framework for APIs
- `uvicorn` - ASGI server
- `streamlit` - Frontend UI framework

### LangChain & LLM
- `langchain` - LLM application framework
- `langchain-community` - Document loaders
- `langchain-ollama` - Ollama integration (embeddings + chat)

### Document Processing
- `pymupdf` - Fast PDF parsing (replaces pypdf)

### Data & Networking
- `numpy` - Vector operations and cosine similarity
- `httpx` - Async HTTP client
- `requests` - Sync HTTP client
- `python-multipart` - File upload handling
- `pydantic` - Data validation

## User Preferences

### Code Style
- Prefer lightweight dependencies
- Optimize for speed and minimal footprint
- Use native Python/numpy when possible over heavy ML libraries

## Important Notes

### Ollama Requirement
This application requires Ollama to be running locally. In Replit environment:
- Ollama is not available by default
- The app will show "Ollama unavailable" warnings
- Document upload and embedding generation will not work without Ollama
- **For production use**: Deploy on a system with Ollama installed

### Supported File Types
- ✅ PDF (via PyMuPDF)
- ❌ TXT (removed for optimization)
- ❌ DOCX (removed for optimization)

To re-enable TXT/DOCX support, add back the loaders in `backend/document_processor.py`.

## Troubleshooting

### Frontend shows "Backend unavailable"
- Check if backend is running: `curl http://localhost:8000/health`
- Restart the Frontend workflow
- Check logs in Replit console

### File uploads fail
- Ensure Ollama is running: `ollama serve`
- Check backend logs for errors
- Verify PDF file is not corrupted

### Deployment issues
- Verify `start_services.sh` is executable
- Check deployment logs in Replit
- Ensure port 5000 is accessible

## File Structure
```
.
├── backend/              # FastAPI backend
│   ├── main.py          # App entry point
│   ├── routes.py        # API endpoints
│   ├── model_manager.py # LLM/embedding manager
│   ├── document_processor.py # PDF processing
│   ├── ollama.py        # Async Ollama client
│   └── ...
├── frontend/            # Streamlit frontend
│   ├── app.py          # Main UI
│   ├── chat.py         # Chat interface
│   ├── sidebar.py      # Document management
│   ├── api_client.py   # Backend API client
│   └── ...
├── vector_store.py      # Custom vector database
├── start_services.sh    # Unified startup script
├── .streamlit/          # Streamlit configuration
│   └── config.toml     # Server settings
├── uploaded_documents/  # PDF storage
└── vector_data/        # Vector embeddings
    └── vectors.json    # Persisted vectors
```

## Version History
- **v1.0** (Oct 28, 2025) - Initial Replit setup with optimized dependencies
