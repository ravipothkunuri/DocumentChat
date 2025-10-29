"""
Data Models for API Requests and Responses

These Pydantic models define what our API expects and returns.
They automatically validate data and generate nice API docs!

Think of them as blueprints for our API's input/output.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    What the user sends when asking a question.
    
    Required:
    - question: What they're asking
    
    Optional stuff:
    - model: Which AI model to use (we have a default)
    - top_k: How many document chunks to search (more = slower but maybe better)
    - temperature: How creative the AI should be
    - stream: Real-time streaming or wait for complete response?
    """
    question: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = None
    top_k: int = Field(4, ge=1, le=20)  # Between 1 and 20
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)  # 0 to 2
    stream: bool = False


class DocumentUploadResponse(BaseModel):
    """
    What we send back after a document upload.
    
    Tells you:
    - Did it work?
    - What was the filename?
    - How many chunks did we create?
    - How big was the file?
    - Any helpful message
    """
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str


class DocumentInfo(BaseModel):
    """
    Info about a document in our system.
    
    Used when listing all documents.
    """
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str


class HealthResponse(BaseModel):
    """
    Health check response - is everything OK?
    
    Just the status and when we checked.
    """
    status: str
    timestamp: str
