# RAG Backend - Overall Architecture Overview

## 🎯 Purpose
A production-ready **Retrieval-Augmented Generation (RAG)** system that allows users to upload documents, stores them as vector embeddings, and enables intelligent question-answering using local LLMs via Ollama.

---

## 🏗️ Core Architecture

### High-Level Flow
```
Documents Upload → Text Processing → Embeddings Generation → Vector Storage → Query → Retrieval → LLM Response
```

---

## 📦 Main Components & Responsibilities

### 1. OllamaLLM Class
**Purpose:** Universal interface for Ollama language models

**Responsibilities:**
- Automatically detects whether a model supports `/api/chat` or `/api/generate` endpoint
- Inspects model metadata to determine correct API format
- Handles both streaming and non-streaming responses
- Manages model warm-up and cold-start timeouts
- Graceful fallback between chat/generate endpoints
- Extracts responses correctly based on endpoint type

**Key Methods:**
- `invoke()` - Single response generation
- `stream()` - Streaming response generation
- `_detect_endpoint()` - Auto-detects correct API endpoint

---

### 2. ConfigManager Class
**Purpose:** Centralized configuration management

**Responsibilities:**
- Loads/saves configuration from `config.json`
- Manages default settings (models, chunk sizes, temperature, etc.)
- Tracks query statistics
- Provides thread-safe config updates
- Persists changes to disk

**Default Configuration:**
```python
{
    'model': 'phi3',
    'embedding_model': 'nomic-embed-text',
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'temperature': 0.7,
    'timeout': 120,
    'cold_start_timeout': 600
}
```

---

### 3. MetadataManager Class
**Purpose:** Document metadata tracking

**Responsibilities:**
- Stores document information (filename, size, chunks, upload date)
- Persists metadata to `metadata.json`
- Provides CRUD operations for document records
- Tracks document processing status
- Enables document listing and validation

**Metadata Structure:**
```python
{
    "filename": "report.pdf",
    "size": 1024000,
    "chunks": 45,
    "status": "processed",
    "uploaded_at": "2025-10-21T10:30:00",
    "type": "pdf"
}
```

---

### 4. ModelManager Class
**Purpose:** LLM and embedding model lifecycle management

**Responsibilities:**
- Lazy initialization of embedding models (only when needed)
- **Pre-warming embeddings model on startup** (prevents first-upload failures)
- Caching LLM instances for reuse
- Testing models with sample inputs before use
- Resetting models when configuration changes
- Managing model temperature and timeout parameters

**Key Features:**
- Embeddings model is tested with "initialization test" query
- LLM models are cached by name for efficiency
- Automatic reinitialization on config changes

---

### 5. DocumentProcessor Class
**Purpose:** Document loading and text chunking

**Responsibilities:**
- Loads different file types (PDF, TXT, DOCX)
- Extracts text content from documents
- Splits text into manageable chunks using RecursiveCharacterTextSplitter
- Adds metadata to each chunk (source, chunk_id, length)
- Configurable chunk size and overlap

**Supported Formats:**
- `.pdf` → PyPDFLoader
- `.txt` → TextLoader
- `.docx` → UnstructuredWordDocumentLoader

---

### 6. VectorStore Class
**Purpose:** Vector similarity search engine (External dependency)

**Responsibilities:**
- Stores document embeddings with metadata
- Performs similarity searches using cosine distance
- Persists vectors to disk
- Manages vector addition/removal
- Provides statistics (chunk count, storage size)

---

## 🔌 API Endpoints

### Document Management

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload` | POST | Upload and process documents into chunks + embeddings |
| `/documents` | GET | List all uploaded documents with metadata |
| `/documents/{filename}` | DELETE | Remove specific document and its vectors |
| `/clear` | DELETE | Remove all documents and vectors |

### Query Operations

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Ask questions about documents (streaming/non-streaming) |

**Query Flow:**
1. Embed user question using embeddings model
2. Search vector store for similar chunks (top_k results)
3. Build context from retrieved chunks
4. Send context + question to LLM
5. Return answer with sources and similarity scores

### Configuration

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/configure` | POST | Update models, chunk settings, temperature |
| `/models` | GET | List available Ollama models |
| `/health` | GET | System health check + configuration status |
| `/stats` | GET | Usage statistics (documents, queries, storage) |

### Maintenance

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rebuild-vectors` | POST | Regenerate all embeddings with current model |

---

## 🔄 Request Flow Examples

### Upload Flow:
```
1. User uploads file.pdf
2. Validate file type and size
3. Save to uploaded_documents/
4. Load document content (PyPDFLoader)
5. Split into chunks (RecursiveCharacterTextSplitter)
6. Generate embeddings for each chunk
7. Store vectors in VectorStore
8. Save metadata to metadata.json
9. Return success with chunk count
```

### Query Flow:
```
1. User asks "What is the main topic?"
2. Generate query embedding
3. Search vector store (cosine similarity)
4. Retrieve top_k similar chunks
5. Build context from chunks
6. Create prompt: context + question
7. Send to LLM (OllamaLLM)
8. Stream or return complete answer
9. Include sources + similarity scores
```

---

## 💾 Data Persistence

### Directory Structure:
```
uploaded_documents/     # Original uploaded files
vector_data/
  ├── metadata.json     # Document metadata
  ├── config.json       # System configuration
  └── vectors.pkl       # Serialized vector embeddings (VectorStore)
rag_app.log            # Application logs
```

---

## 🚀 Startup Process

### Application Lifespan:
```python
1. Initialize managers (Config, Metadata, VectorStore, Models)
2. Create directories if missing
3. Load vector store from disk
4. PRE-INITIALIZE embeddings model (prevents first-upload failure)
5. Start FastAPI server
```

**Critical Fix:** Embeddings model is tested on startup with a sample query to ensure it's ready for immediate use.

---

## 🛡️ Error Handling & Validation

### Upload Validation:
- File type whitelist (pdf, txt, docx)
- Size limit (20MB default)
- Duplicate filename check
- Content extraction validation

### Query Validation:
- Document existence check
- Question length limits (1-5000 chars)
- top_k bounds (1-20)
- Temperature bounds (0.0-2.0)

### Model Error Handling:
- Model not found → suggests `ollama pull`
- Timeout handling with configurable limits
- Automatic endpoint fallback (chat → generate)
- Embeddings validation (checks for empty results)

---

## ⚡ Performance Optimizations

1. **Model Caching:** LLM instances cached by name
2. **Lazy Loading:** Embeddings model only initialized when needed (but pre-warmed on startup)
3. **Streaming Support:** Reduces memory for large responses
4. **Vector Persistence:** Avoids re-embedding on restart
5. **Timeout Management:** Separate cold-start vs warm timeouts

---

## 🔧 Configuration Management

### Dynamic Updates:
- Change models without restart
- Adjust chunk sizes (rebuilds required)
- Modify temperature per query
- Track total query count

### Model Reset Triggers:
- Changing `model` → clears LLM cache
- Changing `embedding_model` → resets embeddings model + warns about rebuild

---

## 📊 Monitoring & Statistics

**Health Check Provides:**
- Ollama availability status
- Current configuration
- Document/chunk counts
- Total queries processed

**Statistics Include:**
- Average chunks per document
- Total storage size
- Vector store size
- Last update timestamp

---

## 🎯 Key Design Decisions

1. **Automatic Endpoint Detection:** No manual configuration needed for chat vs generate models
2. **Pre-initialization:** Prevents cold-start failures on first upload
3. **Metadata Separation:** Keeps file info separate from vector data
4. **Graceful Degradation:** Fallback mechanisms for model detection
5. **CORS Enabled:** Ready for frontend integration
6. **Comprehensive Logging:** Debug-level logging for troubleshooting

---

## 📝 Summary

This architecture provides a robust, production-ready RAG system with:
- ✅ Excellent error handling
- ✅ Automatic model management
- ✅ Flexible configuration options
- ✅ Performance optimizations
- ✅ Comprehensive monitoring
- ✅ Easy integration with frontends

The system is designed to be reliable, maintainable, and scalable for production use! 🚀