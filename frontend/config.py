"""Configuration settings - 15 lines"""
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']
SIDEBAR_WIDTH = 320
DEFAULT_TOP_K = 4

# Fixed model configuration
FIXED_LLM_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
