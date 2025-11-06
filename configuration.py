from pathlib import Path


# Configuration
API_BASE_URL = "http://localhost:8000"
OLLAMA_BASE_URL = "http://localhost:11434"

DEFAULT_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")

# Backend file validation uses suffixes with dots
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}

# UI file uploader uses extensions without dots
UI_ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

THINKING_MESSAGES = [
    "🤔 Analyzing document...",
    "💭 Thinking...",
    "📖 Reading through content...",
    "🔍 Searching for answers...",
    "⚡ Processing your question...",
    "🧠 Understanding the context...",
]
