"""
Configuration Settings for the Backend

This is our central config file - change settings here and they apply everywhere!
Includes paths, model settings, and file upload rules.

All directories get created automatically if they don't exist yet.
"""

import os
from pathlib import Path

# ============================================================================
# API SETTINGS
# ============================================================================

# Where's Ollama running?
# You can override this with an environment variable if needed
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Which AI model should we use?
# llama3.2 is a good balance of speed and quality
FIXED_MODEL = "llama3.2"

# Which model generates our embeddings (text → vectors)?
# nomic-embed-text is fast and works well for most documents
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================================
# WHERE WE STORE STUFF
# ============================================================================

# Uploaded documents go here
UPLOAD_DIR = Path("uploaded_documents")

# Vector store and metadata live here
VECTOR_DIR = Path("vector_data")

# Specific files within VECTOR_DIR
METADATA_FILE = VECTOR_DIR / "metadata.json"    # Info about each document
CONFIG_FILE = VECTOR_DIR / "config.json"        # App settings


# ============================================================================
# FILE UPLOAD RULES
# ============================================================================

# What file types do we accept?
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}

# How big can uploaded files be?
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes


# ============================================================================
# STARTUP: CREATE DIRECTORIES IF NEEDED
# ============================================================================

# Make sure our storage directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)
