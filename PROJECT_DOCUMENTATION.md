# LangChain Ollama RAG Assistant
## Unlocking Innovation - Your Path to AI-Driven Document Intelligence

---

## Scope of the Project

**Project Title:** LangChain Ollama RAG Assistant – AI-powered Document Chat System

### Project Objectives:
- Enable users to upload PDF documents and create a searchable knowledge base
- Implement Retrieval-Augmented Generation (RAG) for accurate, context-aware responses
- Provide real-time streaming chat interface for natural conversations
- Extract and process document content with efficient chunking strategies
- Help users gain instant insights from their document collections

### Deliverables:
- **Streamlit Web Application** for document uploads and chat interface
- **FastAPI Backend** for document processing and query handling
- **PDF Text Extraction** using PyMuPDF (10x faster than PyPDF)
- **Custom Vector Store** with numpy-based cosine similarity
- **LLM Integration** via LangChain with Ollama support
- **Streaming Response System** for real-time chat experience
- **Document Management** with multi-document support and session persistence

---

## Design

### Design Diagram:
```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Frontend  │  HTTP   │   Backend    │  API    │   Ollama    │
│  Streamlit  │◄───────►│   FastAPI    │◄───────►│  LLM Server │
│  (Port 5000)│         │ (Port 8000)  │         │             │
└─────────────┘         └──────────────┘         └─────────────┘
      │                        │
      │                        │
      ▼                        ▼
┌─────────────┐         ┌──────────────┐
│   Session   │         │   Vector     │
│    State    │         │    Store     │
│  (Memory)   │         │   (JSON)     │
└─────────────┘         └──────────────┘
```

### Design Description:

**I. System Architecture:**
The system is designed as a full-stack web application with separated frontend and backend services. Users upload PDF documents through a Streamlit interface, which are processed by a FastAPI backend that extracts text, generates embeddings, stores vectors, and handles chat queries using RAG methodology.

**II. Components:**

**III. Frontend (Streamlit)**
   - A. Document upload and management interface
   - B. Real-time streaming chat with stop functionality
   - C. Session-based chat history per document
   - D. Visual feedback with loading states and notifications
   - E. Minimal custom CSS for optimal performance

**IV. Backend (FastAPI)**
   - A. **Document Processor** - Extracts text from PDFs using PyMuPDF
   - B. **Model Manager** - Manages LLM and embedding models via Ollama
   - C. **Vector Store** - Custom in-memory vector database with JSON persistence
   - D. **API Routes** - RESTful endpoints for upload, query, delete, health checks
   - E. **Async Ollama Client** - Handles streaming responses from LLM

**V. Vector Database**
   - A. Custom implementation with numpy-based cosine similarity
   - B. Persistent storage in JSON format
   - C. Document chunking with configurable overlap
   - D. Metadata tracking for source attribution

**VI. Error Handling:**
   - A. Graceful degradation when Ollama is unavailable
   - B. File validation (size limits, format checks)
   - C. Streaming interruption support
   - D. Clear error messages and warnings

---

## Workflow

**I.** User uploads PDF document via Streamlit UI (`frontend/app.py`)

**II.** Frontend sends file to backend API endpoint (`/upload`)

**III.** Backend extracts text from PDF using PyMuPDF (`backend/document_processor.py`)

**IV.** Text is split into chunks with overlap for better context preservation

**V.** Chunks are sent to Ollama for embedding generation

**VI.** Embeddings and text chunks are stored in vector database (`vector_store.py`)

**VII.** Document metadata is saved for tracking

**VIII.** User selects document and asks a question via chat interface

**IX.** Backend receives query and generates query embedding

**X.** Vector store performs similarity search to find relevant chunks

**XI.** Retrieved chunks are combined with query as context

**XII.** LLM generates streaming response based on context

**XIII.** Frontend displays streaming response with real-time updates

**XIV.** Chat history is maintained per document in session state

**XV.** User can stop generation, clear chat, or delete documents

---

## Test Cases

### Functional Test Cases:

| Test Case ID | Description | Steps | Expected Result |
|--------------|-------------|-------|-----------------|
| TC01 | Upload valid PDF document | Upload a well-formed PDF file | Document processed, embeddings generated, appears in document list |
| TC02 | Upload invalid file (non-PDF) | Upload .txt, .docx, or corrupted file | Error message displayed, file rejected |
| TC03 | Select document and query | Select uploaded document, type question | Relevant response generated from document context |
| TC04 | Streaming response | Ask question and observe response | Text appears incrementally in real-time |
| TC05 | Stop generation | Click stop button during response | Generation stops immediately, partial response saved |
| TC06 | Multi-document support | Upload multiple PDFs | Each document listed separately, independent chat histories |
| TC07 | Delete document | Click delete button on document | Document removed from list and vector store |
| TC08 | Clear chat history | Click "Clear Chat" button | Chat messages cleared, document still accessible |
| TC09 | Backend health check | Access app when backend is down | Error message: "Backend unavailable" |
| TC10 | Ollama unavailable | Query when Ollama is not running | Warning displayed, graceful degradation |

### Edge Test Cases:

| Test Case ID | Description | Steps | Expected Result |
|--------------|-------------|-------|-----------------|
| TC11 | Empty PDF upload | Upload PDF with no extractable text | Warning message about empty content |
| TC12 | Large PDF (100+ pages) | Upload very large PDF document | Processing completes without timeout, chunks created efficiently |
| TC13 | File size limit | Upload file exceeding 20MB limit | Error message about size limit |
| TC14 | Concurrent queries | Send multiple queries rapidly | All queries processed in order, no crashes |
| TC15 | Session persistence | Refresh browser page | Chat history maintained for active session |
| TC16 | No documents uploaded | Access app with no documents | Welcome message displayed with instructions |
| TC17 | Special characters in filename | Upload PDF with special chars in name | File uploaded successfully with sanitized name |
| TC18 | Network interruption | Lose connection during upload | Appropriate error message, no partial uploads |

---

## Tools and Code Details

### Third Party Tools:

| Tool Name | Open Source/Licensed | URL | Purpose |
|-----------|---------------------|-----|---------|
| Ollama (llama3.2) | Open Source | https://ollama.com/library/llama3.2 | Document understanding, embedding generation, and query answering |
| Streamlit | Open Source | https://streamlit.io | Frontend web application framework |
| FastAPI | Open Source | https://fastapi.tiangolo.com | Backend REST API framework |
| LangChain | Open Source | https://langchain.com | LLM application framework and orchestration |
| PyMuPDF | Open Source (AGPL) | https://pymupdf.readthedocs.io | Fast PDF text extraction |
| NumPy | Open Source | https://numpy.org | Vector operations and cosine similarity |

### Technologies Used:

| Technology Name | Version |
|----------------|---------|
| Python | 3.11 |
| AI Framework: LangChain | 0.3.x |
| AI Integration: LangChain-Ollama | Latest |
| Backend Framework: FastAPI | 0.120.1 |
| Frontend Framework: Streamlit | 1.50.0 |
| ASGI Server: Uvicorn | 0.38.0 |
| PDF Processing: PyMuPDF | Latest |
| HTTP Client: HTTPX | Latest |
| Data Validation: Pydantic | Latest |
| Vector Operations: NumPy | Latest |

---

## Project Structure

```
.
├── backend/                    # FastAPI backend
│   ├── main.py                # Application entry point
│   ├── routes.py              # API endpoints
│   ├── model_manager.py       # LLM and embedding management
│   ├── document_processor.py  # PDF processing logic
│   ├── ollama.py              # Async Ollama client
│   ├── config_manager.py      # Configuration management
│   └── models.py              # Pydantic data models
├── frontend/                   # Streamlit frontend
│   ├── app.py                 # Main UI entry point
│   ├── chat.py                # Chat interface component
│   ├── sidebar.py             # Document management UI
│   ├── api_client.py          # Backend API client
│   ├── styles.py              # Minimal custom CSS
│   ├── session_state.py       # Session state management
│   └── toast.py               # Notification system
├── vector_store.py            # Custom vector database
├── start_services.sh          # Unified startup script
├── uploaded_documents/        # PDF storage directory
├── vector_data/               # Vector embeddings storage
│   └── vectors.json           # Persisted vectors
└── .streamlit/                # Streamlit configuration
    └── config.toml            # Server settings
```

---

## Key Features

### 1. **Document Processing**
- Fast PDF extraction using PyMuPDF
- Intelligent text chunking with overlap
- Metadata preservation for source tracking

### 2. **Vector Search**
- Custom in-memory vector store
- Numpy-based cosine similarity (optimized)
- JSON persistence for data durability

### 3. **Chat Interface**
- Real-time streaming responses
- Stop generation capability
- Per-document chat history
- Timestamp tracking

### 4. **Deployment Ready**
- Configured for Replit Autoscale
- Environment-based configuration
- Health check endpoints
- Graceful error handling

---

## Future Enhancements

1. **Multi-format Support**: Add support for TXT, DOCX, and other document formats
2. **Advanced Search**: Implement hybrid search (keyword + semantic)
3. **User Authentication**: Add user accounts and document privacy
4. **Cloud Storage**: Integrate with S3 or cloud storage solutions
5. **Conversation Memory**: Implement multi-turn conversation context
6. **Export Features**: Allow exporting chat histories
7. **Analytics Dashboard**: Show usage statistics and insights

---

## Thank You

### Innovative Technology
### Intelligent Solutions
### Empowered Users

---

*© 2025 LangChain Ollama RAG Assistant. All rights reserved.*
