"""
Pydantic models and LLM client
"""
import json
import logging
import requests
from typing import Dict, Any, Optional, Iterator
from pydantic import BaseModel, Field, validator

from backend.config import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = None
    top_k: int = Field(4, ge=1, le=20)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    stream: bool = False


class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str


class ModelConfig(BaseModel):
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = Field(None, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    
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


class OllamaLLM:
    """Universal Ollama LLM client with automatic endpoint detection"""
    
    CHAT_MODEL_PATTERNS = [
        'llama3', 'llama-3', 'gemma', 'qwen', 'mistral', 'mixtral',
        'phi3', 'phi-3', 'command', 'deepseek', 'llava', 'openchat',
        'solar', 'yi', 'nous', 'dolphin', 'orca', 'vicuna', 'wizardlm'
    ]
    
    def __init__(
        self, 
        model: str, 
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        timeout: int = 120,
        cold_start_timeout: int = 600
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.cold_start_timeout = cold_start_timeout
        self.base_url = base_url.rstrip('/')
        self.endpoint_type = None
        self.model_loaded = False
        self.model_info = None
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Fetch model information from Ollama's show API"""
        if self.model_info is not None:
            return self.model_info
        
        try:
            show_url = f"{self.base_url}/api/show"
            payload = {"name": self.model}
            response = requests.post(show_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.model_info = response.json()
                return self.model_info
            return {}
        except Exception:
            return {}
    
    def _detect_endpoint_from_model_info(self) -> Optional[str]:
        """Detect endpoint type from model information"""
        model_info = self._get_model_info()
        
        if not model_info:
            return None
        
        template = model_info.get('template', '').lower()
        modelfile = model_info.get('modelfile', '').lower()
        
        chat_indicators = [
            '{{.system}}', '{{.prompt}}', '<|im_start|>', '<|start_header_id|>',
            '[inst]', '<|user|>', '<|assistant|>', 'chatml', 'chat_template'
        ]
        
        if any(indicator in template for indicator in chat_indicators):
            return "chat"
        
        if 'chat' in modelfile or any(indicator in modelfile for indicator in chat_indicators):
            return "chat"
        
        parameters = model_info.get('parameters', '')
        if 'chat' in parameters.lower():
            return "chat"
        
        return None
    
    def _detect_endpoint_from_name(self) -> str:
        """Fallback: detect endpoint from model name patterns"""
        model_lower = self.model.lower()
        
        for pattern in self.CHAT_MODEL_PATTERNS:
            if pattern in model_lower:
                return "chat"
        
        return "generate"
    
    def _detect_endpoint(self) -> None:
        """Detect which endpoint the model supports"""
        if self.endpoint_type is not None:
            return
        
        detected_type = self._detect_endpoint_from_model_info()
        
        if detected_type is None:
            detected_type = self._detect_endpoint_from_name()
        
        self.endpoint_type = detected_type
    
    def _verify_endpoint_with_minimal_call(self) -> None:
        """Verify endpoint works with a minimal test call"""
        if self.model_loaded:
            return
        
        try:
            url, payload = self._build_payload("test", stream=False)
            
            if self.endpoint_type == "chat":
                payload["messages"] = [{"role": "user", "content": "Hi"}]
                payload["options"] = {"num_predict": 1}
            else:
                payload["prompt"] = "Hi"
                payload["options"] = {"num_predict": 1}
            
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                self.model_loaded = True
            elif response.status_code == 404 and self.endpoint_type == "chat":
                self.endpoint_type = "generate"
                self.model_loaded = False
        except Exception:
            pass
    
    def _build_payload(self, prompt: str, stream: bool) -> tuple[str, dict]:
        """Build request payload based on endpoint type"""
        self._detect_endpoint()
        
        common_options = {"temperature": self.temperature}
        
        if self.endpoint_type == "chat":
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "options": common_options
            }
        else:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": common_options
            }
        
        return url, payload
    
    def _extract_content(self, data: dict) -> str:
        """Extract content from response based on endpoint type"""
        if self.endpoint_type == "chat":
            return data.get("message", {}).get("content", "")
        return data.get("response", "")

    def _get_timeout(self, is_first_call: bool = False) -> int:
        """Get the timeout value"""
        if is_first_call or not self.model_loaded:
            return self.cold_start_timeout
        return self.timeout

    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt"""
        current_timeout = self._get_timeout()
        try:
            if not self.model_loaded:
                self._verify_endpoint_with_minimal_call()
            
            url, payload = self._build_payload(prompt, stream=False)

            response = requests.post(url, json=payload, timeout=current_timeout)
            
            if response.status_code == 404 and self.endpoint_type == "chat":
                self.endpoint_type = "generate"
                url, payload = self._build_payload(prompt, stream=False)
                response = requests.post(url, json=payload, timeout=current_timeout)
            
            response.raise_for_status()
            self.model_loaded = True
            return self._extract_content(response.json())
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Model error: {str(e)}")
        except requests.exceptions.Timeout:
            raise ValueError(f"Request timed out after {current_timeout}s")
        except Exception as e:
            raise ValueError(f"Failed to communicate with model: {str(e)}")
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the model's response"""
        response = None
        current_timeout = self._get_timeout()
        try:
            if not self.model_loaded:
                self._verify_endpoint_with_minimal_call()
            
            url, payload = self._build_payload(prompt, stream=True)
            
            response = requests.post(url, json=payload, timeout=current_timeout, stream=True)
            
            if response.status_code == 404 and self.endpoint_type == "chat":
                self.endpoint_type = "generate"
                url, payload = self._build_payload(prompt, stream=True)
                response = requests.post(url, json=payload, timeout=current_timeout, stream=True)
            
            response.raise_for_status()
            first_chunk = True
            
            try:
                for line in response.iter_lines(decode_unicode=True, chunk_size=1):
                    if line:
                        try:
                            data = json.loads(line)
                            content = self._extract_content(data)
                            
                            if first_chunk and content:
                                self.model_loaded = True
                                first_chunk = False

                            if content:
                                yield content
                            
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            except GeneratorExit:
                logger.info("Stream generator closed by client")
                if response is not None:
                    try:
                        response.close()
                    except:
                        pass
                raise
                    
        except GeneratorExit:
            logger.info("Stream interrupted by client disconnect")
            if response is not None:
                try:
                    response.close()
                except:
                    pass
            raise
        except requests.exceptions.Timeout:
            error_msg = f"Streaming request timed out after {current_timeout}s. Please try again."
            logger.error(error_msg)
            raise ValueError(error_msg)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Streaming error: {str(e)}")
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            raise ValueError(f"Failed to stream: {str(e)}")
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
