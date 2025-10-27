"""Utility Functions - 35 lines"""
import re
import httpx
from pathlib import Path
from fastapi import HTTPException
from backend.config import OLLAMA_BASE_URL, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB

def clean_llm_response(text: str) -> str:
    """Remove reasoning tags from LLM response"""
    patterns = [
        r'<think>.*?</think>',
        r'<reasoning>.*?</reasoning>',
        r'</?(?:think|reasoning)>'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def check_ollama_health() -> tuple[bool, str]:
    """Check if Ollama is available"""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200, "Available" if response.status_code == 200 else "Not responding"
    except Exception as e:
        return False, f"Error: {str(e)}"

def validate_file(filename: str, file_size: int):
    """Validate uploaded file"""
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size/(1024*1024):.1f}MB > {MAX_FILE_SIZE_MB}MB)"
        )
