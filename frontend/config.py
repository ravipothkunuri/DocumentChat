"""
Configuration settings for the RAG application
"""
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# File Upload Settings
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']

# UI Settings
SIDEBAR_WIDTH = 320
DEFAULT_TOP_K = 4

# Streaming & Heartbeat Settings
# Frontend timeout should be at least 2x the backend heartbeat interval
# to allow for network latency and processing delays
DEFAULT_HEARTBEAT_INTERVAL = 10  # Backend sends heartbeat every 10s by default
STREAM_TIMEOUT = 300  # Frontend timeout (5 minutes)

# Model Defaults
DEFAULT_LLM_MODEL = "phi3"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_MODELS = {
    "ollama": {
        "llm_models": ["phi3", "llama3", "mistral", "deepseek-r1"],
        "embedding_models": ["nomic-embed-text"]
    },
    "current_config": {
        "model": DEFAULT_LLM_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL
    }
}
