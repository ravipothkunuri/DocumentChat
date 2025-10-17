import os
import json
import logging
import traceback
import subprocess
import re
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Iterator, Literal
from contextlib import asynccontextmanager
from enum import Enum

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain_ollama import OllamaEmbeddings

from vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
METADATA_FILE = VECTOR_DIR / "metadata.json"
CONFIG_FILE = VECTOR_DIR / "config.json"
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Model Provider Enum
class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


# Pydantic Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="The question to ask")
    model: Optional[str] = Field(None, description="LLM model to use")
    provider: Optional[ModelProvider] = Field(None, description="Model provider")
    top_k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature")
    stream: bool = Field(False, description="Stream response in real-time")


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: int
    similarity_scores: List[float]
    processing_time: float
    model_used: str
    provider: str


class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str


class ModelConfig(BaseModel):
    model: Optional[str] = None
    provider: Optional[ModelProvider] = None
    embedding_model: Optional[str] = None
    embedding_provider: Optional[ModelProvider] = None
    chunk_size: Optional[int] = Field(None, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if v is not None and 'chunk_size' in values and values['chunk_size'] is not None:
            if v >= values['chunk_size']:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class DocumentInfo(BaseModel):
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str


# Base LLM Interface
class BaseLLM:
    """Base class for all LLM providers."""
    
    def __init__(self, model: str, temperature: float = 0.7, timeout: int = 120, **kwargs):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.kwargs = kwargs
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt and return the complete response."""
        raise NotImplementedError
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response."""
        raise NotImplementedError


# Universal Ollama LLM with automatic endpoint detection
class OllamaLLM(BaseLLM):
    """Universal Ollama LLM client with automatic endpoint detection."""
    
    def __init__(
        self, 
        model: str, 
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.endpoint_type = None
        
        logger.info(f"Initializing OllamaLLM with model: {model}")
    
    def _lazy_detect_endpoint(self) -> None:
        """Lazily detect which endpoint the model supports (called on first use)."""
        if self.endpoint_type is not None:
            return  # Already detected
        
        logger.debug(f"Detecting endpoint for model: {self.model}")
        
        # Try chat endpoint first
        try:
            chat_url = f"{self.base_url}/api/chat"
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False
            }
            
            response = requests.post(chat_url, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                self.endpoint_type = "chat"
                logger.info(f"Model {self.model} supports /api/chat endpoint")
                return
        except Exception as e:
            logger.debug(f"Chat endpoint test failed: {e}")
        
        # Try generate endpoint
        try:
            generate_url = f"{self.base_url}/api/generate"
            test_payload = {
                "model": self.model,
                "prompt": "hi",
                "stream": False
            }
            
            response = requests.post(generate_url, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                self.endpoint_type = "generate"
                logger.info(f"Model {self.model} supports /api/generate endpoint")
                return
        except Exception as e:
            logger.debug(f"Generate endpoint test failed: {e}")
        
        # Default to generate if both fail (will error on actual use)
        logger.warning(f"Could not detect endpoint for {self.model}, defaulting to generate")
        self.endpoint_type = "generate"
    
    def _call_chat_endpoint(self, prompt: str, stream: bool = False) -> requests.Response:
        """Call the /api/chat endpoint."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "options": {"temperature": self.temperature}
        }
        
        return requests.post(url, json=payload, timeout=self.timeout, stream=stream)
    
    def _call_generate_endpoint(self, prompt: str, stream: bool = False) -> requests.Response:
        """Call the /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": self.temperature}
        }
        
        return requests.post(url, json=payload, timeout=self.timeout, stream=stream)
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt and return the complete response."""
        self._lazy_detect_endpoint()
        
        try:
            if self.endpoint_type == "chat":
                response = self._call_chat_endpoint(prompt, stream=False)
            else:
                response = self._call_generate_endpoint(prompt, stream=False)
            
            response.raise_for_status()
            data = response.json()
            
            if self.endpoint_type == "chat":
                return data.get("message", {}).get("content", "")
            else:
                return data.get("response", "")
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Model {self.model} returned error: {str(e)}")
        except requests.exceptions.Timeout:
            raise ValueError(f"Model {self.model} request timed out after {self.timeout}s")
        except Exception as e:
            raise ValueError(f"Failed to communicate with model {self.model}: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response."""
        self._lazy_detect_endpoint()
        
        try:
            if self.endpoint_type == "chat":
                response = self._call_chat_endpoint(prompt, stream=True)
            else:
                response = self._call_generate_endpoint(prompt, stream=True)
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        
                        if self.endpoint_type == "chat":
                            content = data.get("message", {}).get("content", "")
                        else:
                            content = data.get("response", "")
                        
                        if content:
                            yield content
                        
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Model {self.model} streaming error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to stream from model {self.model}: {str(e)}")


# OpenAI LLM
class OpenAILLM(BaseLLM):
    """OpenAI API compatible LLM client."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = (api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip('/')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        logger.info(f"Initializing OpenAILLM with model: {model}")
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to communicate with OpenAI: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": True
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise ValueError(f"Failed to stream from OpenAI: {str(e)}")


# Anthropic LLM
class AnthropicLLM(BaseLLM):
    """Anthropic Claude API client."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.api_base = "https://api.anthropic.com/v1"
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        logger.info(f"Initializing AnthropicLLM with model: {model}")
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt."""
        url = f"{self.api_base}/messages"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 4096
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to communicate with Anthropic: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response."""
        url = f"{self.api_base}/messages"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 4096,
            "stream": True
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "content_block_delta":
                                content = data.get("delta", {}).get("text", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise ValueError(f"Failed to stream from Anthropic: {str(e)}")


# Google Gemini LLM
class GoogleLLM(BaseLLM):
    """Google Gemini API client."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, **kwargs)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.api_base = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        logger.info(f"Initializing GoogleLLM with model: {model}")
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt."""
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature}
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Google API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to communicate with Google: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response."""
        url = f"{self.api_base}/models/{self.model}:streamGenerateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature}
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if "candidates" in data:
                            content = data["candidates"][0]["content"]["parts"][0].get("text", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise ValueError(f"Failed to stream from Google: {str(e)}")


# Cohere LLM
class CohereLLM(BaseLLM):
    """Cohere API client."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, **kwargs)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.api_base = "https://api.cohere.ai/v1"
        
        if not self.api_key:
            raise ValueError("Cohere API key is required")
        
        logger.info(f"Initializing CohereLLM with model: {model}")
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt."""
        url = f"{self.api_base}/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["generations"][0]["text"]
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Cohere API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to communicate with Cohere: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response."""
        url = f"{self.api_base}/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": True
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if data.get("text"):
                            yield data["text"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise ValueError(f"Failed to stream from Cohere: {str(e)}")


# Hugging Face LLM
class HuggingFaceLLM(BaseLLM):
    """Hugging Face Inference API client."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, **kwargs)
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.api_base = "https://api-inference.huggingface.co/models"
        
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")
        
        logger.info(f"Initializing HuggingFaceLLM with model: {model}")
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt."""
        url = f"{self.api_base}/{self.model}"
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": self.temperature}
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            return str(data)
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Hugging Face API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to communicate with Hugging Face: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response (not widely supported, falls back to invoke)."""
        # Most HF models don't support streaming, so we fall back to chunked response
        result = self.invoke(prompt)
        # Simulate streaming by yielding in chunks
        chunk_size = 50
        for i in range(0, len(result), chunk_size):
            yield result[i:i + chunk_size]


# Configuration Manager with provider support
class ConfigManager:
    """Centralized configuration management."""
    
    DEFAULT_CONFIG = {
        'model': 'phi3',
        'provider': 'ollama',
        'embedding_model': 'nomic-embed-text',
        'embedding_provider': 'ollama',
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'temperature': 0.7,
        'total_queries': 0,
        'api_keys': {}
    }
    
    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                self.config.update(loaded_config)
                logger.info(f"Loaded config: model={self.config['model']}, provider={self.config['provider']}")
            else:
                logger.info("No existing config file, using defaults")
                self.save()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug("Configuration saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update(self, **kwargs) -> List[str]:
        """Update configuration and return list of changed fields."""
        changed = []
        for key, value in kwargs.items():
            if value is not None:
                if key == 'api_key':
                    # Store API keys separately
                    provider = kwargs.get('provider', self.config.get('provider'))
                    if 'api_keys' not in self.config:
                        self.config['api_keys'] = {}
                    self.config['api_keys'][provider] = value
                    changed.append('api_key')
                elif key in self.config or key in ['provider', 'embedding_provider', 'api_base']:
                    if self.config.get(key) != value:
                        self.config[key] = value
                        changed.append(key)
        
        if changed:
            self.save()
        
        return changed
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        return self.config.get('api_keys', {}).get(provider)
    
    def increment_queries(self) -> None:
        """Increment query counter."""
        self.config['total_queries'] += 1
        self.save()


# Metadata Manager
class MetadataManager:
    """Manage document metadata."""
    
    def __init__(self, metadata_file: Path = METADATA_FILE):
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self) -> None:
        """Load metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} documents")
            else:
                self.metadata = {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}
    
    def save(self) -> None:
        """Save metadata to file."""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add(self, filename: str, metadata: Dict[str, Any]) -> None:
        """Add or update document metadata."""
        self.metadata[filename] = metadata
        self.save()
    
    def remove(self, filename: str) -> bool:
        """Remove document metadata."""
        if filename in self.metadata:
            del self.metadata[filename]
            self.save()
            return True
        return False
    
    def get(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document metadata."""
        return self.metadata.get(filename)
    
    def exists(self, filename: str) -> bool:
        """Check if document exists."""
        return filename in self.metadata
    
    def clear(self) -> None:
        """Clear all metadata."""
        self.metadata = {}
        self.save()
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all document metadata."""
        return list(self.metadata.values())


# Universal Model Manager
class ModelManager:
    """Manage LLM and embedding models across all providers."""
    
    # Provider to LLM class mapping
    PROVIDER_MAP = {
        ModelProvider.OLLAMA: OllamaLLM,
        ModelProvider.OPENAI: OpenAILLM,
        ModelProvider.ANTHROPIC: AnthropicLLM,
        ModelProvider.GOOGLE: GoogleLLM,
        ModelProvider.COHERE: CohereLLM,
        ModelProvider.HUGGINGFACE: HuggingFaceLLM,
    }
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.embeddings_model: Optional[OllamaEmbeddings] = None
        self.llm_cache: Dict[str, BaseLLM] = {}
    
    def get_embeddings_model(self) -> OllamaEmbeddings:
        """Get or create embeddings model."""
        if self.embeddings_model is None:
            model_name = self.config.get('embedding_model')
            provider = self.config.get('embedding_provider', 'ollama')
            
            logger.info(f"Initializing embeddings model: {model_name} ({provider})")
            
            try:
                # Currently only Ollama embeddings supported
                # Can be extended to support OpenAI, Cohere, etc.
                self.embeddings_model = OllamaEmbeddings(
                    model=model_name,
                    base_url=OLLAMA_BASE_URL
                )
                
                # Test the model
                test_embedding = self.embeddings_model.embed_query("test")
                logger.info(f"Embeddings model '{model_name}' initialized, dimensions: {len(test_embedding)}")
            except Exception as e:
                logger.error(f"Error initializing embeddings model '{model_name}': {e}")
                raise ValueError(f"Failed to initialize embeddings model '{model_name}': {str(e)}")
        
        return self.embeddings_model
    
    def get_llm_model(
        self, 
        model_name: Optional[str] = None,
        provider: Optional[ModelProvider] = None
    ) -> BaseLLM:
        """Get or create LLM model (with caching)."""
        model_to_use = model_name or self.config.get('model')
        provider_to_use = provider or ModelProvider(self.config.get('provider', 'ollama'))
        
        cache_key = f"{provider_to_use}:{model_to_use}"
        
        # Check cache
        if cache_key not in self.llm_cache:
            logger.info(f"Initializing LLM model: {model_to_use} ({provider_to_use})")
            
            try:
                llm_class = self.PROVIDER_MAP.get(provider_to_use)
                if not llm_class:
                    raise ValueError(f"Unsupported provider: {provider_to_use}")
                
                # Prepare initialization kwargs
                init_kwargs = {
                    'model': model_to_use,
                    'temperature': self.config.get('temperature')
                }
                
                # Add provider-specific parameters
                if provider_to_use == ModelProvider.OLLAMA:
                    init_kwargs['base_url'] = OLLAMA_BASE_URL
                else:
                    # For cloud providers, add API key
                    api_key = self.config.get_api_key(provider_to_use.value)
                    if api_key:
                        init_kwargs['api_key'] = api_key
                    
                    # Add custom API base if configured
                    api_base = self.config.get('api_base')
                    if api_base and provider_to_use == ModelProvider.OPENAI:
                        init_kwargs['api_base'] = api_base
                
                llm = llm_class(**init_kwargs)
                self.llm_cache[cache_key] = llm
                logger.info(f"LLM model '{model_to_use}' ({provider_to_use}) cached successfully")
            except Exception as e:
                logger.error(f"Error initializing LLM model '{model_to_use}' ({provider_to_use}): {e}")
                raise ValueError(f"Failed to initialize LLM model '{model_to_use}': {str(e)}")
        
        return self.llm_cache[cache_key]
    
    def reset_embeddings_model(self) -> None:
        """Reset embeddings model."""
        self.embeddings_model = None
        logger.debug("Embeddings model reset")
    
    def reset_llm_cache(self) -> None:
        """Reset LLM cache."""
        self.llm_cache.clear()
        logger.debug("LLM cache cleared")
    
    def update_temperature(self, temperature: float) -> None:
        """Update LLM temperature for all cached models."""
        for llm in self.llm_cache.values():
            llm.temperature = temperature


# Document Processor
class DocumentProcessor:
    """Handle document loading and processing."""
    
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': UnstructuredWordDocumentLoader
    }
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def load_document(self, file_path: Path) -> List[str]:
        """Load document content based on file type."""
        suffix = file_path.suffix.lower()
        
        if suffix not in self.LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        loader_class = self.LOADERS[suffix]
        loader = loader_class(str(file_path))
        documents = loader.load()
        content = [doc.page_content for doc in documents]
        
        logger.debug(f"Loaded {len(content)} pages from {file_path}")
        return content
    
    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        """Split text content into chunks."""
        full_text = "\n\n".join(content)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size'),
            chunk_overlap=self.config.get('chunk_overlap'),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(full_text)
        
        documents = [
            {
                "page_content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "chunk_length": len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
        logger.info(f"Split {filename} into {len(documents)} chunks")
        return documents


# Utility Functions
def clean_llm_response(text: str) -> str:
    """Clean LLM response by removing reasoning tags."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?(?:think|reasoning|thought)>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_available_models() -> tuple[List[str], List[str]]:
    """Get available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'].split(':')[0] for model in data.get('models', [])]
            
            embedding_models = [m for m in models if 'embed' in m.lower() or 'nomic' in m.lower()]
            llm_models = [m for m in models if 'embed' not in m.lower()]
            
            return llm_models, embedding_models
    except Exception as e:
        logger.debug(f"API-based model detection failed: {e}")
    
    return ["phi3", "llama3", "mistral"], ["nomic-embed-text"]


def check_ollama_available(config: ConfigManager) -> tuple[bool, str]:
    """Check if Ollama is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code != 200:
            return False, "Ollama server not responding"
        
        return True, f"Ollama available with {config.get('embedding_model')} and {config.get('model')}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def validate_file(filename: str, file_size: int) -> None:
    """Validate uploaded file."""
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size / (1024*1024):.1f} MB) exceeds max {MAX_FILE_SIZE_MB} MB"
        )


# Initialize global objects
config_manager = ConfigManager()
metadata_manager = MetadataManager()
vector_store = VectorStore()
model_manager = ModelManager(config_manager)
document_processor = DocumentProcessor(config_manager)

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting RAG Application...")
    
    try:
        vector_store.load()
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load vector store: {e}")
    
    logger.info("RAG Application started successfully")
    
    yield
    
    logger.info("Shutting down RAG Application...")


# Initialize FastAPI app
app = FastAPI(
    title="Universal RAG Assistant API",
    description="Production-ready RAG system with multi-provider LLM support",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.get("/health", tags=["Health"])
async def health_check():
    """Check system health and configuration."""
    try:
        ollama_available, ollama_message = check_ollama_available(config_manager)
        stats = vector_store.get_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ollama_status": {
                "available": ollama_available,
                "message": ollama_message
            },
            "configuration": {
                "model": config_manager.get('model'),
                "provider": config_manager.get('provider'),
                "embedding_model": config_manager.get('embedding_model'),
                "embedding_provider": config_manager.get('embedding_provider'),
                "chunk_size": config_manager.get('chunk_size'),
                "chunk_overlap": config_manager.get('chunk_overlap'),
                "temperature": config_manager.get('temperature')
            },
            "document_count": len(metadata_manager.metadata),
            "total_chunks": stats.get("total_chunks", 0),
            "total_queries": config_manager.get('total_queries')
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    logger.info(f"Upload request for file: {file.filename}")
    
    content = await file.read()
    file_size = len(content)
    
    validate_file(file.filename, file_size)
    
    if metadata_manager.exists(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists"
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
        
        document_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(document_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        embeddings_model = model_manager.get_embeddings_model()
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        vector_store.add_documents(documents, embeddings)
        
        metadata_manager.add(file.filename, {
            "filename": file.filename,
            "size": file_size,
            "chunks": len(documents),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        })
        
        vector_store.save()
        
        logger.info(f"Processed {file.filename}: {len(documents)} chunks, {file_size} bytes")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=file_size,
            message=f"Document processed successfully into {len(documents)} chunks"
        )
    except Exception as e:
        logger.error(f"Error processing upload {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        if file_path.exists():
            file_path.unlink()
        metadata_manager.remove(file.filename)
        
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


# @app.post("/query", tags=["Query"])
# async def query_documents(request: QueryRequest):
#     """Query the document collection."""
#     model_info = request.model or config_manager.get('model')
#     provider_info = request.provider or config_manager.get('provider')
    
#     logger.info(f"Query request: '{request.question[:50]}...' with {model_info} ({provider_info})")
    
#     start_time = datetime.now()
    
#     try:
#         # Validate documents exist
#         stats = vector_store.get_stats()
#         if stats.get("total_chunks", 0) == 0:
#             raise HTTPException(
#                 status_code=400,
#                 detail="No documents available. Please upload documents first."
#             )
        
#         # Get embeddings model
#         try:
#             embeddings_model = model_manager.get_embeddings_model()
#         except ValueError as e:
#             logger.error(f"Embeddings model error: {e}")
#             raise HTTPException(status_code=503, detail=str(e))
        
#         # Generate query embedding
#         try:
#             query_embedding = embeddings_model.embed_query(request.question)
#         except Exception as e:
#             logger.error(f"Embedding generation failed: {e}")
#             raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {str(e)}")
        
#         # Search similar documents
#         similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
#         if not similar_docs:
#             raise HTTPException(status_code=400, detail="No relevant documents found")
        
#         # Prepare context
#         context_parts = []
#         sources = []
#         similarity_scores = []
        
#         for doc, score in similar_docs:
#             context_parts.append(f"Source: {doc['metadata']['source']}\nContent: {doc['page_content']}")
#             sources.append(doc['metadata']['source'])
#             similarity_scores.append(float(score))
        
#         context = "\n\n".join(context_parts)
        
#         # Get LLM
#         try:
#             llm = model_manager.get_llm_model(request.model, request.provider)
#             if request.temperature is not None:
#                 llm.temperature = request.temperature
#         except ValueError as e:
#             logger.error(f"LLM model error: {e}")
#             raise HTTPException(status_code=503, detail=str(e))
        
#         # Build prompt
#         unique_sources = list(set(sources))
#         doc_identity = unique_sources[0] if len(unique_sources) == 1 else f"your documents"
        
#         prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person.

# Your content includes:
# {context}

# The user asks: {request.question}

# Respond naturally as the document. Your response:"""
        
#         # Stream or regular response
#         if request.stream:
#             async def generate() -> AsyncGenerator[str, None]:
#                 try:
#                     metadata = {
#                         "sources": sources,
#                         "chunks_used": len(similar_docs),
#                         "similarity_scores": similarity_scores,
#                         "model_used": llm.model,
#                         "provider": provider_info,
#                         "type": "metadata"
#                     }
#                     yield f"data: {json.dumps(metadata)}\n\n"
                    
#                     for chunk in llm.stream(prompt):
#                         if len(chunk) > 0:
#                             yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                    
#                     processing_time = (datetime.now() - start_time).total_seconds()
#                     completion = {"type": "done", "processing_time": processing_time}
#                     yield f"data: {json.dumps(completion)}\n\n"
                    
#                     config_manager.increment_queries()
#                     logger.info(f"Streaming query completed in {processing_time:.2f}s")
                
#                 except Exception as e:
#                     logger.error(f"Streaming error: {e}")
#                     error_msg = {"type": "error", "error": str(e)}
#                     yield f"data: {json.dumps(error_msg)}\n\n"
            
#             return StreamingResponse(generate(), media_type="text/event-stream")
        
#         else:
#             # Regular response
#             try:
#                 answer = llm.invoke(prompt)
#                 answer = clean_llm_response(answer)
#             except ValueError as e:
#                 logger.error(f"LLM invocation failed: {e}")
#                 raise HTTPException(status_code=503, detail=str(e))
            
#             processing_time = (datetime.now() - start_time).total_seconds()
#             config_manager.increment_queries()
            
#             logger.info(f"Query completed in {processing_time:.2f}s, {len(similar_docs)} chunks used")
            
#             return QueryResponse(
#                 answer=answer,
#                 sources=sources,
#                 chunks_used=len(similar_docs),
#                 similarity_scores=similarity_scores,
#                 processing_time=processing_time,
#                 model_used=llm.model,
#                 provider=provider_info
#             )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         logger.debug(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")



@app.post("/query", tags=["Query"])
async def query_documents(request: Request, query: QueryRequest):
    """Query the document collection with client disconnect handling."""
    logger.info(f"Query request: '{query.question[:50]}...' with model {query.model or config_manager.get('model')}")
    start_time = datetime.now()
    try:
        # Validate documents exist
        stats = vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents available. Please upload documents first."
            )
        
        # Get embeddings model
        try:
            embeddings_model = model_manager.get_embeddings_model()
        except ValueError as e:
            logger.error(f"Embeddings model error: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        
        # Generate query embedding
        try:
            query_embedding = embeddings_model.embed_query(query.question)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {str(e)}")
        
        # Search similar documents
        similar_docs = vector_store.similarity_search(query_embedding, k=query.top_k)
        if not similar_docs:
            raise HTTPException(status_code=400, detail="No relevant documents found")
        
        # Prepare context
        context_parts = []
        sources = []
        similarity_scores = []
        for doc, score in similar_docs:
            context_parts.append(f"Source: {doc['metadata']['source']}\nContent: {doc['page_content']}")
            sources.append(doc['metadata']['source'])
            similarity_scores.append(float(score))
        context = "\n\n".join(context_parts)
        
        # Get LLM
        try:
            llm = model_manager.get_llm_model(query.model)
            if query.temperature is not None:
                llm.temperature = query.temperature
        except ValueError as e:
            logger.error(f"LLM model error: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        
        # Build prompt
        unique_sources = list(set(sources))
        doc_identity = unique_sources[0] if len(unique_sources) == 1 else "your documents"
        prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person.

Your content includes:
{context}

The user asks: {query.question}

Respond naturally as the document. Your response:"""
        
        # Streaming or regular response
        if query.stream:
            async def generate():
                try:
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": similarity_scores,
                        "model_used": llm.model,
                        "type": "metadata"
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"

                    for chunk in llm.stream(prompt):
                        # Abort if client disconnected
                        if await request.is_disconnected():
                            logger.info("Client disconnected -- stopping streaming response.")
                            break
                        if len(chunk) > 0:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                    processing_time = (datetime.now() - start_time).total_seconds()
                    completion = {"type": "done", "processing_time": processing_time}
                    yield f"data: {json.dumps(completion)}\n\n"
                    config_manager.increment_queries()
                    logger.info(f"Streaming query completed in {processing_time:.2f}s")
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_msg = {"type": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_msg)}\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            try:
                answer = llm.invoke(prompt)
                answer = clean_llm_response(answer)
            except ValueError as e:
                logger.error(f"LLM invocation failed: {e}")
                raise HTTPException(status_code=503, detail=str(e))
            processing_time = (datetime.now() - start_time).total_seconds()
            config_manager.increment_queries()
            logger.info(f"Query completed in {processing_time:.2f}s with model '{llm.model}'")
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(similar_docs),
                "similarity_scores": similarity_scores,
                "processing_time": processing_time,
                "model_used": llm.model
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")



@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all uploaded documents."""
    logger.debug("Listing documents")
    documents = [DocumentInfo(**meta) for meta in metadata_manager.list_all()]
    logger.info(f"Listed {len(documents)} documents")
    return documents


@app.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a specific document."""
    logger.info(f"Delete request for document: {filename}")
    
    if not metadata_manager.exists(filename):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    
    try:
        vector_store.remove_documents_by_source(filename)
        metadata_manager.remove(filename)
        
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        vector_store.save()
        
        logger.info(f"Successfully deleted document: {filename}")
        return {
            "message": f"Document '{filename}' deleted successfully",
            "note": "Vector store has been updated automatically"
        }
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.delete("/clear", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents and embeddings."""
    logger.info("Clear all documents request")
    
    try:
        vector_store.clear()
        metadata_manager.clear()
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        vector_store.save()
        
        logger.info("Successfully cleared all documents")
        return {"message": "All documents and embeddings cleared successfully", "cleared": True}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@app.get("/documents/{filename}/preview", tags=["Documents"])
async def preview_document(filename: str, num_chunks: int = Query(3, ge=1, le=10)):
    """Preview document chunks."""
    logger.debug(f"Preview request for {filename}, {num_chunks} chunks")
    
    if not metadata_manager.exists(filename):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    
    try:
        documents = vector_store.get_documents_by_source(filename)
        
        if not documents:
            raise HTTPException(status_code=404, detail=f"No chunks found for document '{filename}'")
        
        preview_chunks = documents[:num_chunks]
        
        chunks_data = [
            {
                "chunk_id": doc['metadata']['chunk_id'],
                "content": doc['page_content'][:500] + "..." if len(doc['page_content']) > 500 else doc['page_content'],
                "length": len(doc['page_content'])
            }
            for doc in preview_chunks
        ]
        
        logger.debug(f"Preview generated for {filename}: {len(chunks_data)} chunks")
        
        return {
            "filename": filename,
            "total_chunks": len(documents),
            "preview_chunks": len(chunks_data),
            "chunks": chunks_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating preview for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")


@app.post("/configure", tags=["Configuration"])
async def configure_system(config_update: ModelConfig):
    """Update system configuration."""
    logger.info(f"Configuration update request: {config_update.dict(exclude_none=True)}")
    
    try:
        changed_fields = config_manager.update(**config_update.dict(exclude_none=True))
        
        if "model" in changed_fields or "provider" in changed_fields:
            model_manager.reset_llm_cache()
        
        if "embedding_model" in changed_fields or "embedding_provider" in changed_fields:
            model_manager.reset_embeddings_model()
        
        if "temperature" in changed_fields:
            model_manager.update_temperature(config_manager.get('temperature'))
        
        logger.info(f"Configuration updated successfully, changed fields: {changed_fields}")
        
        response = {"message": "Configuration updated successfully", "changed_fields": changed_fields}
        
        if "embedding_model" in changed_fields or "embedding_provider" in changed_fields:
            response["warning"] = "Embedding model changed. Consider rebuilding vectors for existing documents."
        
        return response
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@app.get("/models", tags=["Configuration"])
async def get_models():
    """Get available models and providers."""
    logger.debug("Get models request")
    
    try:
        llm_models, embedding_models = get_available_models()
        
        if not llm_models:
            llm_models = ["phi3", "llama3", "mistral"]
        if not embedding_models:
            embedding_models = ["nomic-embed-text"]
        
        return {
            "providers": [p.value for p in ModelProvider],
            "ollama": {
                "llm_models": llm_models,
                "embedding_models": embedding_models,
            },
            "current_config": {
                "model": config_manager.get('model'),
                "provider": config_manager.get('provider'),
                "embedding_model": config_manager.get('embedding_model'),
                "embedding_provider": config_manager.get('embedding_provider')
            }
        }
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {
            "providers": [p.value for p in ModelProvider],
            "ollama": {
                "llm_models": ["phi3", "llama3", "mistral"],
                "embedding_models": ["nomic-embed-text"],
            },
            "current_config": {
                "model": config_manager.get('model'),
                "provider": config_manager.get('provider'),
                "embedding_model": config_manager.get('embedding_model'),
                "embedding_provider": config_manager.get('embedding_provider')
            }
        }


@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get system statistics."""
    logger.debug("Statistics request")
    
    try:
        stats = vector_store.get_stats()
        
        total_size = sum(meta.get('size', 0) for meta in metadata_manager.metadata.values())
        doc_count = len(metadata_manager.metadata)
        avg_chunks = stats.get('total_chunks', 0) / max(1, doc_count)
        
        return {
            "total_documents": doc_count,
            "total_chunks": stats.get('total_chunks', 0),
            "total_queries": config_manager.get('total_queries'),
            "total_storage_size": total_size,
            "average_chunks_per_document": round(avg_chunks, 2),
            "last_update": stats.get('last_update'),
            "vector_store_size_mb": stats.get('vector_store_size', 0) / (1024 * 1024) if stats.get('vector_store_size') else 0
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "total_storage_size": 0,
            "average_chunks_per_document": 0.0,
            "last_update": None,
            "vector_store_size_mb": 0.0
        }


@app.post("/rebuild-vectors", tags=["Debug"])
async def rebuild_vectors():
    """Rebuild all vectors with current embedding model."""
    logger.info("Rebuild vectors request")
    
    try:
        vector_store.clear()
        results = {}
        
        for filename in list(metadata_manager.metadata.keys()):
            try:
                logger.debug(f"Rebuilding vectors for {filename}")
                
                file_path = UPLOAD_DIR / filename
                if not file_path.exists():
                    results[filename] = {"success": False, "error": "File not found"}
                    continue
                
                document_content = document_processor.load_document(file_path)
                documents = document_processor.split_text(document_content, filename)
                
                embeddings_model = model_manager.get_embeddings_model()
                texts = [doc["page_content"] for doc in documents]
                embeddings = embeddings_model.embed_documents(texts)
                
                vector_store.add_documents(documents, embeddings)
                
                metadata = metadata_manager.get(filename) or {}
                metadata["chunks"] = len(documents)
                metadata_manager.add(filename, metadata)
                
                results[filename] = {"success": True, "chunks": len(documents)}
                logger.debug(f"Successfully rebuilt {filename}: {len(documents)} chunks")
            except Exception as e:
                logger.error(f"Error rebuilding {filename}: {e}")
                results[filename] = {"success": False, "error": str(e)}
        
        vector_store.save()
        
        success_count = sum(1 for result in results.values() if result["success"])
        
        logger.info(f"Rebuild completed: {success_count}/{len(results)} documents successful")
        
        return {
            "message": f"Rebuild completed: {success_count}/{len(results)} documents processed successfully",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error rebuilding vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild vectors: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Universal RAG Backend Server on port 8000...")
    uvicorn.run(
        "rag_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
