# 📚 Chat With Documents using AI

A powerful, async-capable RAG (Retrieval-Augmented Generation) application that lets you upload documents and have intelligent conversations with them using local AI models via Ollama.

## 🌟 Features

### Core Functionality
- **📄 Multi-Format Support**: Upload PDF, TXT, and DOCX files
- **💬 Per-Document Chat**: Separate conversation history for each document
- **⚡ Real-Time Streaming**: See AI responses as they're generated
- **🎯 Smart Retrieval**: Vector-based semantic search for relevant content
- **🛑 Stop Generation**: Interrupt AI responses anytime
- **📤 Export Conversations**: Save chats as JSON or Markdown

### Technical Highlights
- **Async Architecture**: Non-blocking operations for smooth UX
- **Persistent Storage**: Automatic saving of documents, embeddings, and metadata
- **Intelligent Chunking**: Smart document splitting with overlap for context
- **Vector Store**: Custom in-memory vector database with JSON persistence
- **Lazy Loading**: Models initialized only when needed
- **Health Monitoring**: Real-time system status and statistics

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │  (Frontend)
│  - Chat Interface
│  - File Upload
│  - Export Tools
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI API   │  (Backend)
│  - Document Mgmt
│  - Query Handler
│  - Streaming
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌──────┐ ┌─────────┐
│Vector │ │ Model │ │Config│ │Metadata │
│ Store │ │Manager│ │ Mgr  │ │   Mgr   │
└───────┘ └───────┘ └──────┘ └─────────┘
    │         │
    │         ▼
    │    ┌────────┐
    │    │ Ollama │
    │    │  LLM   │
    │    └────────┘
    │
    ▼
┌─────────────┐
│   Uploads   │
│ + JSON Data │
└─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9 or higher
- **Ollama**: Installed and running locally
- **Required Models**:
  - `llama3.2` (LLM for responses)
  - `nomic-embed-text` (for embeddings)

### Installation

1. **Install Ollama** (if not already installed):
   ```bash
   # Visit https://ollama.com for installation
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull Required Models**:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. **Clone Repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

4. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start Backend (Terminal 1)**:
   ```bash
   python -m backend.main
   ```
   Backend will start on `http://localhost:8000`

2. **Start Frontend (Terminal 2)**:
   ```bash
   streamlit run app.py
   ```
   Frontend will open at `http://localhost:8501`

3. **Start Chatting**:
   - Upload a document (PDF/TXT/DOCX)
   - Select it from the sidebar
   - Ask questions!

## 📦 Dependencies

### Backend
```
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-community==0.0.13
langchain-ollama==0.0.1
langchain-text-splitters==0.0.1
httpx==0.25.2
pymupdf==1.23.8
docx2txt==0.8
numpy==1.24.3
pydantic==2.5.2
```

### Frontend
```
streamlit==1.29.0
httpx==0.25.2
```

## 🗂️ Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application entry
│   ├── routes.py            # API endpoints (240 lines)
│   ├── managers.py          # Config, Metadata, Model managers
│   ├── ollama.py            # Async Ollama client
│   ├── utils.py             # Document processing utilities
│   ├── models.py            # Pydantic request/response models
│   └── config.py            # Backend configuration
│
├── frontend/
│   ├── app.py               # Streamlit main entry
│   ├── chat.py              # Chat interface with streaming
│   ├── sidebar.py           # Document management UI
│   ├── api_client.py        # Async API client
│   ├── utils.py             # Frontend helpers
│   ├── styles.py            # Custom CSS
│   └── config.py            # Frontend configuration
│
├── vector_store.py          # Custom vector database
├── uploads/                 # Uploaded documents (auto-created)
├── vector_data/             # Vector embeddings (auto-created)
└── config/                  # App configuration (auto-created)
```

## 🎯 Key Components

### Backend

#### **ConfigManager**
- Saves/loads app settings automatically
- Tracks query statistics
- Persists configurations across restarts

#### **MetadataManager**
- Tracks all uploaded documents
- Stores file info (size, chunks, timestamps)
- JSON-based persistence

#### **ModelManager**
- Lazy loading of AI models (memory efficient)
- Caches embeddings and LLM models
- Proper cleanup on shutdown

#### **VectorStore**
- Custom in-memory vector database
- Cosine similarity search
- JSON persistence for embeddings

#### **AsyncOllamaLLM**
- Async streaming responses
- Connection pooling
- Error handling and timeouts

### Frontend

#### **Chat Interface**
- Real-time streaming responses
- Per-document conversation history
- Stop generation capability
- Export to JSON/Markdown

#### **Document Management**
- Multi-file upload support
- Auto-selection of uploaded docs
- Delete functionality
- File size validation

## 🔧 Configuration

### Backend (`backend/config.py`)
```python
OLLAMA_BASE_URL = "http://localhost:11434"
FIXED_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
```

### Frontend (`config.py`)
```python
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']
```

## 📊 API Endpoints

### Health Check
```http
GET /health
```
Returns system status, document count, and Ollama availability.

### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

Response: {
  "status": "success",
  "filename": "document.pdf",
  "chunks": 45,
  "file_size": 524288,
  "message": "Document processed into 45 chunks"
}
```

### Query Documents
```http
POST /query
Content-Type: application/json

{
  "question": "What is this about?",
  "top_k": 4,
  "temperature": 0.7,
  "stream": true
}
```

### List Documents
```http
GET /documents
```

### Delete Document
```http
DELETE /documents/{filename}
```

### Clear All
```http
DELETE /clear
```

### Statistics
```http
GET /stats
```

## 💡 Usage Tips

### Document Chunking
- **Chunk Size**: 1000 characters (configurable)
- **Overlap**: 200 characters (maintains context)
- **Strategy**: Recursive splitting (paragraphs → sentences → words)

### Best Practices
1. **Document Quality**: Clear, well-formatted documents work best
2. **Question Clarity**: Be specific in your questions
3. **Context**: The AI sees only the most relevant chunks (top 4 by default)
4. **Model Selection**: llama3.2 balances speed and quality

### Streaming Responses
- Responses appear in real-time as the AI generates them
- Click the ⏹️ button to stop generation anytime
- Interrupted responses are marked in chat history

## 🔒 Security Considerations

- Files validated for type and size before processing
- No authentication implemented (add if deploying publicly)
- API runs on localhost by default
- Consider adding rate limiting for production use

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve
```

### Models not found
```bash
# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text

# Verify models are available
ollama list
```

### Upload fails
- Check file size (max 20MB by default)
- Verify file format (PDF/TXT/DOCX only)
- Ensure `uploads/` directory is writable

### Slow responses
- First query initializes models (takes ~5-10 seconds)
- Subsequent queries are much faster
- Consider using a smaller model for speed
- Check your system resources (Ollama needs RAM)

## 🎨 Customization

### Change AI Model
Edit `backend/config.py`:
```python
FIXED_MODEL = "llama2"  # or any Ollama model
```

### Adjust Chunk Size
Edit `backend/managers.py`:
```python
DEFAULT_CONFIG = {
    'chunk_size': 1500,      # Increase for more context
    'chunk_overlap': 300,    # Increase with chunk_size
}
```

### Modify Temperature
Edit query request:
```python
QueryRequest(
    question="...",
    temperature=0.3  # Lower = more focused, Higher = more creative
)
```

## 📈 Performance

- **Document Processing**: ~2-5 seconds per document
- **Query Response**: ~1-3 seconds (streaming starts immediately)
- **Memory Usage**: ~2GB RAM (includes Ollama models)
- **Storage**: Embeddings are ~4KB per chunk

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add authentication/authorization
- [ ] Support more file formats (Excel, PowerPoint)
- [ ] Implement conversation memory
- [ ] Add multiple model selection
- [ ] Create Docker deployment
- [ ] Add unit tests

## 🙏 Acknowledgments

- **Ollama**: Local LLM inference
- **LangChain**: LLM framework
- **Streamlit**: Frontend framework
- **FastAPI**: Backend framework

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review Ollama documentation: https://ollama.com/docs

---

**Built with ❤️ using Ollama, FastAPI, and Streamlit**
