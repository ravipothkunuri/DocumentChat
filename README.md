# LangChain Ollama RAG Assistant 📚

A production-ready Retrieval-Augmented Generation (RAG) system powered by LangChain, Ollama local LLMs, and Streamlit. Upload documents, generate embeddings using local Ollama models, and query your knowledge base with intelligent, context-aware responses.

## Features

- **LangChain Integration**: Built with LangChain framework for robust RAG pipeline
- **Document Processing**: Upload PDF, TXT, and DOCX files using LangChain document loaders
- **Local LLM Integration**: Uses Ollama models for embeddings and chat (no API keys required)
- **Custom Vector Store**: In-memory vector database with JSON persistence and sklearn cosine similarity
- **Streaming Responses**: Real-time LLM output display with LangChain streaming
- **Background Processing**: Async document upload handling with FastAPI BackgroundTasks
- **Responsive Design**: Mobile-friendly Streamlit UI with adaptive layout
- **Production-Ready**: Complete RAG implementation with proper error handling and logging

## Architecture

### Backend (FastAPI)
- **rag_backend.py**: Main API server with document processing and query endpoints
- **vector_store.py**: Custom vector database implementation
- Runs on: `http://localhost:8000`

### Frontend (Streamlit)
- **app.py**: Interactive UI with document management and querying
- Runs on: `http://localhost:5000`

## Dependencies

### Core Framework Dependencies
- `fastapi>=0.119.0` - Modern web framework for building APIs
- `uvicorn>=0.37.0` - ASGI server for FastAPI
- `streamlit>=1.50.0` - Frontend UI framework

### LangChain & LLM Dependencies
- `langchain>=0.3.27` - Framework for LLM applications and RAG pipelines
- `langchain-community>=0.3.31` - Community document loaders (PDF, TXT, DOCX)
- `langchain-ollama>=0.3.10` - Ollama integration for embeddings and chat
- `ollama>=0.6.0` - Python client for Ollama local LLMs

### Data Processing & Machine Learning
- `numpy>=2.3.4` - Numerical computing library
- `scikit-learn>=1.7.2` - Machine learning library (used for cosine similarity)

### Document Processing
- `pypdf>=6.1.1` - PDF document parsing
- `python-docx>=1.2.0` - DOCX document parsing
- `docx2txt>=0.9` - Additional DOCX text extraction

### API & Networking
- `requests>=2.32.5` - HTTP library for API calls
- `python-multipart>=0.0.20` - File upload handling

### Data Validation
- `pydantic>=2.12.2` - Data validation and settings management

### Standard Library Modules (Built-in)
These are included with Python and don't need installation:
- `json` - JSON encoding/decoding
- `os` - Operating system interfaces
- `time` - Time access and conversions
- `logging` - Logging facility
- `datetime` - Date and time operations
- `pathlib` - Object-oriented filesystem paths
- `typing` - Type hints support
- `traceback` - Exception stack trace utilities

## Installation

### Prerequisites
1. **Python 3.11+** installed
2. **Ollama** installed and running locally
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull required models:
   ollama pull nomic-embed-text  # Embedding model
   ollama pull llama3            # Default LLM model
   ollama pull mistral           # Alternative LLM
   ollama pull phi               # Lightweight LLM
   ```

### Install Dependencies
All dependencies are managed in `pyproject.toml`. Install using:

```bash
# Using pip
pip install fastapi uvicorn streamlit langchain langchain-community langchain-ollama ollama numpy scikit-learn pypdf docx2txt requests python-multipart pydantic

# Or if you have uv or poetry, they'll read from pyproject.toml automatically
```

## Running the Application

### Start Backend Server
```bash
python -m uvicorn rag_backend:app --host 0.0.0.0 --port 8000
```

### Start Frontend UI
```bash
streamlit run app.py --server.port 5000
```

Access the application at: `http://localhost:5000`

## Usage

### 1. Dashboard
- View system health status
- Check available Ollama models
- See document statistics

### 2. Upload Documents
- Upload PDF, TXT, or DOCX files
- Documents are automatically processed and embedded
- View and manage uploaded documents
- Delete documents when no longer needed

### 3. Configuration
- Select LLM model (llama3, mistral, phi)
- Select embedding model (nomic-embed-text, etc.)
- Adjust response parameters:
  - Top K: Number of relevant chunks to retrieve (1-20)
  - Temperature: Response randomness (0.0-1.0)
  - Max Tokens: Maximum response length

### 4. Query Your Documents (via API)
Query endpoint: `POST /query`
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What is the main topic of my documents?",
    "top_k": 5,
    "stream": True
})
```

## API Endpoints

### Health Check
- `GET /health` - Check system status

### Configuration
- `GET /models` - List available Ollama models and current configuration
- `POST /configure` - Update system configuration (LLM model, embedding model, temperature, etc.)

### Documents
- `GET /documents` - List uploaded documents
- `POST /upload` - Upload document (supports PDF, TXT, DOCX)
- `DELETE /documents/{filename}` - Delete specific document
- `DELETE /clear` - Clear all documents

### Query
- `POST /query` - Query documents with streaming support

## Configuration

### Streamlit Configuration
Located in `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Vector Store
- Embeddings stored in memory with JSON persistence
- Storage file: `vector_store.json`
- Automatic save on document changes

## Mobile Support

The application includes responsive CSS for:
- Mobile devices (≤768px)
- Tablets (769-1024px)
- Touch-friendly interfaces (44px minimum tap targets)
- Adaptive sidebar behavior
- Full-width buttons and form elements on mobile

## Project Structure

```
.
├── app.py                 # Streamlit frontend
├── rag_backend.py         # FastAPI backend
├── vector_store.py        # Custom vector database
├── pyproject.toml         # Dependency management
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── uploads/               # Uploaded documents (created at runtime)
└── vector_store.json      # Vector embeddings (created at runtime)
```

## How It Works

1. **Document Upload**: Files are uploaded and processed in the background
2. **Text Chunking**: Documents are split into manageable chunks using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Each chunk is embedded using Ollama's embedding model
4. **Vector Storage**: Embeddings are stored in the custom vector store with metadata
5. **Query Processing**: User questions are embedded and matched against stored vectors using cosine similarity
6. **Context Retrieval**: Top-K most relevant chunks are retrieved
7. **LLM Response**: Retrieved context is sent to the LLM to generate an answer
8. **Streaming Display**: Response is streamed in real-time to the UI

## Troubleshooting

### Backend Won't Start
- Ensure Ollama is running: `ollama serve`
- Check if port 8000 is available
- Verify Python 3.11+ is installed

### Models Not Found
- Pull required models: `ollama pull nomic-embed-text`
- Verify Ollama is accessible: `ollama list`

### Upload Failures
- Check file format (PDF, TXT, DOCX only)
- Ensure file size is reasonable
- Check backend logs for errors

## License

This project is provided as-is for educational and production use.
