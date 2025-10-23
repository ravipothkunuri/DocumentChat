"""
Utility functions
"""
import re
import requests
from typing import List, Tuple
from pathlib import Path
from fastapi import HTTPException

from backend.config import OLLAMA_BASE_URL, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES


def clean_llm_response(text: str) -> str:
    """Clean LLM response by removing reasoning tags"""
    patterns = [
        r'<think>.*?</think>',
        r'<reasoning>.*?</reasoning>',
        r'<thought>.*?</thought>',
        r'</?(?:think|reasoning|thought)>'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_available_models() -> Tuple[List[str], List[str]]:
    """Get available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            all_models = [model['name'] for model in data.get('models', [])]
            
            embedding_models = [m for m in all_models if 'embed' in m.lower() or 'nomic' in m.lower()]
            llm_models = [m for m in all_models if m not in embedding_models]
            
            return llm_models, embedding_models
    except Exception:
        pass
    
    return ["phi3", "llama3", "mistral", "deepseek-r1"], ["nomic-embed-text"]


def check_ollama_health() -> Tuple[bool, str]:
    """Check if Ollama is available"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200, "Ollama available" if response.status_code == 200 else "Ollama not responding"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def validate_file(filename: str, file_size: int) -> None:
    """Validate uploaded file"""
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_BYTES / 1024 / 1024}MB"
        )
