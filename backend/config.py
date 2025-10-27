"""Backend Configuration - 22 lines"""
import os
from pathlib import Path

# API Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
FIXED_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# File Storage
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
METADATA_FILE = VECTOR_DIR / "metadata.json"
CONFIG_FILE = VECTOR_DIR / "config.json"

# File Upload Constraints
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)
