"""
Pydantic models and LLM client
"""
import json
import logging
import httpx
from typing import Dict, Any, Optional, AsyncIterator
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
    """Universal Ollama LLM client with dynamic endpoint detection via show API"""
    
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
    
    async def _get_model_info(self) -> Dict[str, Any]:
        """Fetch model information from Ollama's show API"""
        if self.model_info is not None:
            return self.model_info
        
        try:
            show_url = f"{self.base_url}/api/show"
            payload = {"name": self.model}
            async with httpx.AsyncClient() as client:
                response = await client.post(show_url, json=payload, timeout=10)
            
                if response.status_code == 200:
                    self.model_info = response.json()
                    return self.model_info
            return {}
        except Exception:
            return {}
    
    async def _detect_endpoint_from_model_info(self) -> Optional[str]:
        """Detect endpoint type dynamically from Ollama's show API"""
        model_info = await self._get_model_info()
        
        if not model_info:
            logger.debug(f"No model info available for {self.model}, will try chat endpoint first")
            return None
        
        template = model_info.get('template', '').lower()
        modelfile = model_info.get('modelfile', '').lower()
        parameters = model_info.get('parameters', '').lower()
        
        # Chat API indicators in the model template/configuration
        chat_indicators = [
            # Common chat template tokens
            '{{.system}}', '{{.prompt}}', '{{.messages}}',
            # Popular chat formats
            '<|im_start|>', '<|im_end|>',  # ChatML format
            '<|start_header_id|>', '<|end_header_id|>',  # Llama 3
            '[inst]', '[/inst]',  # Mistral/Mixtral
            '<|user|>', '<|assistant|>',  # Generic chat roles
            # Chat template indicators
            'chatml', 'chat_template', 'conversation',
            # Role-based templates
            '### instruction', '### response',
            'user:', 'assistant:',
        ]
        
        # Check template for chat indicators
        if any(indicator in template for indicator in chat_indicators):
            logger.debug(f"Detected chat endpoint from template for {self.model}")
            return "chat"
        
        # Check modelfile for chat configuration
        if any(indicator in modelfile for indicator in chat_indicators):
            logger.debug(f"Detected chat endpoint from modelfile for {self.model}")
            return "chat"
        
        # Check parameters for chat hints
        if 'chat' in parameters or 'conversation' in parameters:
            logger.debug(f"Detected chat endpoint from parameters for {self.model}")
            return "chat"
        
        # If we have detailed model info but no chat indicators, it's likely a generate model
        if template or modelfile:
            logger.debug(f"No chat indicators found for {self.model}, using generate endpoint")
            return "generate"
        
        # No conclusive information
        return None
    
    async def _detect_endpoint(self) -> None:
        """Detect which endpoint the model supports using show API"""
        if self.endpoint_type is not None:
            return
        
        # Try to detect from model info (show API)
        detected_type = await self._detect_endpoint_from_model_info()
        
        # If we couldn't determine from show API, default to chat
        # Modern models typically support chat, and we'll verify with a test call
        if detected_type is None:
            logger.debug(f"Could not determine endpoint from model info, defaulting to chat for {self.model}")
            detected_type = "chat"
        
        self.endpoint_type = detected_type
    
    async def _verify_endpoint_with_minimal_call(self) -> None:
        """Verify endpoint works with a minimal test call"""
        if self.model_loaded:
            return
        
        try:
            url, payload = await self._build_payload("test", stream=False)
            
            if self.endpoint_type == "chat":
                payload["messages"] = [{"role": "user", "content": "Hi"}]
                payload["options"] = {"num_predict": 1}
            else:
                payload["prompt"] = "Hi"
                payload["options"] = {"num_predict": 1}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=15)
            
                if response.status_code == 200:
                    self.model_loaded = True
                elif response.status_code == 404 and self.endpoint_type == "chat":
                    self.endpoint_type = "generate"
                    self.model_loaded = False
        except Exception:
            pass
    
    async def _build_payload(self, prompt: str, stream: bool) -> tuple[str, dict]:
        """Build request payload based on endpoint type"""
        await self._detect_endpoint()
        
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

    async def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt"""
        current_timeout = self._get_timeout()
        try:
            if not self.model_loaded:
                await self._verify_endpoint_with_minimal_call()
            
            url, payload = await self._build_payload(prompt, stream=False)

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=current_timeout)
            
                if response.status_code == 404 and self.endpoint_type == "chat":
                    self.endpoint_type = "generate"
                    url, payload = await self._build_payload(prompt, stream=False)
                    response = await client.post(url, json=payload, timeout=current_timeout)
            
                response.raise_for_status()
                self.model_loaded = True
                return self._extract_content(response.json())
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Model error: {str(e)}")
        except httpx.TimeoutException:
            raise ValueError(f"Request timed out after {current_timeout}s")
        except Exception as e:
            raise ValueError(f"Failed to communicate with model: {str(e)}")
    
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream the model's response with optimized timeout handling"""
        
        # Separate connect timeout (fast fail) from read timeout (patient streaming)
        # Connect timeout: 30s to establish connection
        # Read timeout: None to allow slow token generation from large models
        connect_timeout = 30
        read_timeout = None  # Infinite read timeout for streaming
        timeout = httpx.Timeout(connect_timeout, read=read_timeout)
        
        try:
            if not self.model_loaded:
                await self._verify_endpoint_with_minimal_call()
            
            url, payload = await self._build_payload(prompt, stream=True)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    
                    if response.status_code == 404 and self.endpoint_type == "chat":
                        self.endpoint_type = "generate"
                        url, payload = await self._build_payload(prompt, stream=True)
                        async with client.stream("POST", url, json=payload) as response:
                            response.raise_for_status()
                            first_chunk = True
                            
                            try:
                                async for line in response.aiter_lines():
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
                                raise
                    else:
                        response.raise_for_status()
                        first_chunk = True
                        
                        try:
                            async for line in response.aiter_lines():
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
                            raise
                    
        except GeneratorExit:
            logger.info("Stream interrupted by client disconnect")
            raise
        except httpx.TimeoutException as e:
            # Connection timeout (not read timeout since read_timeout=None)
            error_msg = f"Failed to connect to Ollama within {connect_timeout}s. Please check if Ollama is running."
            logger.error(f"{error_msg} Error: {e}")
            raise ValueError(error_msg)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found. Pull it using: ollama pull {self.model}")
            raise ValueError(f"Streaming error: {str(e)}")
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            raise ValueError(f"Failed to stream: {str(e)}")
