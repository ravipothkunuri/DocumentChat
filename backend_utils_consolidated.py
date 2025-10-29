"""
Backend Helper Functions

This file has all the utility functions that make our backend work smoothly:
- Cleaning up messy LLM responses
- Checking if Ollama is actually running
- Making sure uploaded files are safe
- Loading and chunking documents properly

Think of it as the toolbox for our backend!
"""

import re
import httpx
from pathlib import Path
from typing import List, Dict, Any
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from backend.config import (
    OLLAMA_BASE_URL, ALLOWED_EXTENSIONS, 
    MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_llm_response(text: str) -> str:
    """
    Strip out the "thinking out loud" parts from AI responses.
    
    Some AI models like to show their reasoning process wrapped in tags like
    <think> or <reasoning>. While that's interesting for debugging, users
    don't need to see it. This function removes all that noise!
    
    Also cleans up excessive blank lines because nobody likes those.
    
    Example:
        Before: "<think>Let me calculate...</think>The answer is 42\n\n\n"
        After:  "The answer is 42"
    """
    # All the tag patterns we want to remove
    patterns = [
        r'<think>.*?</think>',         # <think> blocks
        r'<reasoning>.*?</reasoning>',  # <reasoning> blocks
        r'</?(?:think|reasoning)>'      # Any leftover tags
    ]
    
    # Zap 'em all!
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace 3+ newlines with just 2 (keeps paragraphs readable)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def check_ollama_health() -> tuple[bool, str]:
    """
    Quick health check: Is Ollama actually running?
    
    We ping Ollama's API to see if it's alive. This helps us show a friendly
    error message if the service is down, instead of cryptic connection errors.
    
    Returns:
        (True, "Available") if all good
        (False, "Error: <reason>") if something's wrong
    
    Usage:
        is_healthy, message = check_ollama_health()
        if not is_healthy:
            print(f"Uh oh: {message}")
    """
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        is_ok = response.status_code == 200
        message = "Available" if is_ok else f"Got HTTP {response.status_code}"
        return is_ok, message
    except httpx.TimeoutException:
        return False, "Error: Took too long to respond"
    except httpx.ConnectError:
        return False, "Error: Can't connect (is Ollama running?)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def validate_file(filename: str, file_size: int):
    """
    Make sure uploaded files are safe and not too huge.
    
    We check two things:
    1. File type is allowed (no .exe files, thanks!)
    2. File isn't massive (we have limits for a reason)
    
    Raises an HTTP error if validation fails, with a helpful message.
    
    Example:
        validate_file("report.pdf", 5_000_000)  # ✓ All good
        validate_file("virus.exe", 1000)        # ✗ Raises HTTPException
    """
    file_path = Path(filename)
    
    # Check if we support this file type
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ', '.join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Can't handle {file_path.suffix} files. Try: {allowed}"
        )
    
    # Check if the file is too big
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File is {size_mb:.1f}MB but max is {MAX_FILE_SIZE_MB}MB"
        )


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """
    Loads documents and chops them into digestible chunks.
    
    The chunking strategy is important! We want chunks that are:
    - Big enough to have context
    - Small enough to be focused
    - Overlapping slightly so we don't lose info at boundaries
    
    This class handles all that magic for you.
    """
    
    # Map file types to their loader classes
    LOADERS = {
        '.pdf': PyMuPDFLoader
        # Easy to add more later:
        # '.txt': TextLoader,
        # '.docx': Docx2txtLoader,
    }
    
    def __init__(self, config):
        """Set up the processor with our chunking configuration."""
        self.config = config
    
    def load_document(self, file_path: Path) -> List[str]:
        """
        Extract all the text from a document file.
        
        We pick the right loader based on the file extension, then extract
        everything into a list of strings (one per page or section).
        
        Returns:
            List of text strings from the document
            
        Raises:
            ValueError if we don't support that file type
        
        Example:
            pages = processor.load_document(Path("report.pdf"))
            print(f"Got {len(pages)} pages of text!")
        """
        suffix = file_path.suffix.lower()
        
        if suffix not in self.LOADERS:
            supported = ', '.join(self.LOADERS.keys())
            raise ValueError(
                f"Don't know how to handle {suffix} files. "
                f"We support: {supported}"
            )
        
        # Use the right loader for this file type
        loader_class = self.LOADERS[suffix]
        loader = loader_class(str(file_path))
        documents = loader.load()
        
        # Pull out just the text content
        return [doc.page_content for doc in documents]
    
    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        """
        Chop the document into smart, overlapping chunks.
        
        We use RecursiveCharacterTextSplitter which is fancy because it:
        - Tries to split on paragraph boundaries first
        - Falls back to sentences, then words if needed
        - Keeps some overlap so context doesn't get lost
        
        Each chunk gets wrapped with metadata so we can track where it came from.
        
        Args:
            content: List of text strings (like pages)
            filename: Name of the source file
            
        Returns:
            List of dicts with 'page_content' and 'metadata' keys
        
        Example:
            chunks = processor.split_text(["Page 1...", "Page 2..."], "doc.pdf")
            print(f"Made {len(chunks)} chunks from the document")
        """
        # Combine all pages with blank lines between them
        full_text = "\n\n".join(content)
        
        # Set up our intelligent text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size', 1000),
            chunk_overlap=self.config.get('chunk_overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Try these in order
        )
        
        # Do the splitting
        chunks = splitter.split_text(full_text)
        
        # Wrap each chunk with its metadata
        return [
            {
                "page_content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i  # So we can keep them in order
                }
            }
            for i, chunk in enumerate(chunks)
        ]
