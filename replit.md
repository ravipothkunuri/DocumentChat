# RAG Document Assistant

## Overview

This is a Retrieval-Augmented Generation (RAG) application that enables users to upload documents (PDF, TXT, DOCX), process them into searchable chunks, and query them using local Large Language Models through Ollama. The system consists of a FastAPI backend that handles document processing, embedding generation, and question-answering, paired with a Streamlit frontend for user interaction.

The application uses a custom in-memory vector store with JSON persistence for storing document embeddings, eliminating the need for external vector databases. It leverages cosine similarity for semantic search and integrates with Ollama for both embedding generation (nomic-embed-text) and language model inference (llama3, mistral, phi, etc.).

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Architecture Pattern
- **Separation of Concerns**: The application follows a clear separation between frontend (Streamlit), backend API (FastAPI), and data storage (custom vector store)
- **Rationale**: This modular approach allows independent scaling, testing, and maintenance of each component
- **Alternative Considered**: Monolithic Streamlit application - rejected due to limitations in API reusability and scalability

### Frontend Layer (Streamlit)
- **Technology Choice**: Streamlit for rapid UI development
- **Responsibilities**: User interface, file uploads, query interface, document management
- **Communication**: HTTP requests to FastAPI backend at `localhost:8000`
- **Pros**: Fast prototyping, built-in components, Python-native
- **Cons**: Limited customization compared to React/Vue frameworks

### Backend API Layer (FastAPI)
- **RESTful Design**: Implements standard CRUD operations for document management
- **Key Endpoints**:
  - `POST /upload` - Document upload and processing
  - `DELETE /documents/{filename}` - Individual document deletion
  - `DELETE /clear` - Clear all documents
  - `GET /documents` - List all documents with metadata
  - `GET /documents/{filename}/preview` - Preview document chunks
  - `POST /query` - Question-answering with RAG
  - `GET /health` - Health check endpoint
- **Async Processing**: Background tasks for document processing to avoid blocking requests
- **Error Handling**: Comprehensive exception handling with cleanup mechanisms

### Document Processing Pipeline
- **Text Splitting Strategy**: RecursiveCharacterTextSplitter from LangChain
  - Chunk size: 1000 characters (configurable)
  - Overlap: 200 characters (configurable)
  - Rationale: Balances context preservation with retrieval precision
- **Document Loaders**: Format-specific loaders (PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader)
- **Metadata Enrichment**: Each chunk includes source filename and chunk ID for traceability

### Vector Storage Solution
- **Custom In-Memory Vector Store**: Implemented in `vector_store.py`
- **Persistence**: JSON-based storage for embeddings and metadata
- **Storage Location**: `vector_data/vectors.json`
- **Rationale**: Eliminates external database dependencies, simplifies deployment
- **Trade-offs**: 
  - Pros: Zero setup, portable, version-controllable
  - Cons: Not suitable for massive datasets, limited to single-instance deployments
- **Alternative Considered**: Pinecone/Weaviate - rejected to maintain local-first architecture

### Embedding & Retrieval Strategy
- **Embedding Model**: nomic-embed-text via Ollama
- **Similarity Search**: Cosine similarity using scikit-learn
- **Retrieval Configuration**: Top-k chunks (default: 4, range: 1-20)
- **Dimensions**: Automatically detected from embedding model output
- **Rationale**: Cosine similarity is computationally efficient and effective for semantic search

### Language Model Integration
- **LLM Provider**: Ollama (local inference)
- **Supported Models**: llama3, mistral, phi, and other Ollama-compatible models
- **Streaming Support**: Real-time response streaming capability
- **Temperature Control**: Configurable creativity parameter (0.0-2.0)
- **Rationale**: Local LLMs provide privacy, cost efficiency, and no external API dependencies

### Query Processing Flow
1. **Question Embedding**: Convert user query to vector using same embedding model
2. **Similarity Search**: Find top-k most relevant document chunks
3. **Context Assembly**: Combine retrieved chunks into context
4. **LLM Prompting**: Send context + question to language model
5. **Response Generation**: Stream or return complete answer with source attribution
6. **Metadata Tracking**: Return similarity scores, processing time, sources used

### Data Persistence
- **Vector Data**: JSON files in `vector_data/` directory
- **Document Metadata**: Stored alongside embeddings with timestamps
- **Logging**: Dual logging to console and `rag_app.log` file
- **Cleanup Strategy**: Automatic cleanup on document deletion and errors

### Error Handling & Resilience
- **Backend Health Checks**: `/health` endpoint for monitoring
- **File Validation**: Type checking before processing
- **Duplicate Detection**: Prevents reprocessing same filenames
- **Graceful Degradation**: Error responses with detailed messages
- **Resource Cleanup**: Ensures temporary files are removed on failure

## External Dependencies

### LLM Infrastructure
- **Ollama**: Local LLM server running on default port
  - Required for: Embedding generation and text generation
  - Models needed: nomic-embed-text (embeddings), llama3/mistral/phi (inference)
  - Must be running before application starts

### Python Libraries
- **FastAPI**: Web framework for REST API
- **Uvicorn**: ASGI server for FastAPI
- **Streamlit**: Frontend framework
- **LangChain**: Document processing and LLM abstractions
  - `langchain-community`: Document loaders
  - `langchain-ollama`: Ollama integrations
- **Requests**: HTTP client for frontend-to-backend communication
- **NumPy**: Numerical operations for embeddings
- **scikit-learn**: Cosine similarity calculations
- **Pydantic**: Request/response validation

### Document Processing
- **PyPDF2/pdfplumber** (via PyPDFLoader): PDF parsing
- **python-docx** (via UnstructuredWordDocumentLoader): DOCX parsing
- **Text files**: Native Python support

### Storage & File System
- **Local File System**: All data stored locally
- **JSON**: Serialization format for vector storage
- **Pathlib**: Cross-platform path handling

### CORS Configuration
- **Middleware**: Enabled for cross-origin requests
- **Rationale**: Allows Streamlit frontend to communicate with FastAPI backend on different ports