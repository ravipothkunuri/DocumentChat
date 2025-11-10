from pathlib import Path

# Configuration
FALLBACK_BASE_URL = "http://localhost:8000"
OLLAMA_BASE_URL = "http://localhost:11434"
UI_ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']
DEFAULT_MODEL = "llama3.2"
THINKING_MESSAGES = ["🤔 Analyzing document...", "💭 Thinking...", "📖 Reading through content...", 
                     "🔍 Searching for answers...", "⚡ Processing your question...", "🧠 Understanding the context..."]

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024