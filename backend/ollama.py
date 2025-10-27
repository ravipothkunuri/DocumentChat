"""Async Ollama LLM Client - 60 lines"""
import json
import httpx
from typing import AsyncIterator
from backend.config import OLLAMA_BASE_URL, FIXED_MODEL

class AsyncOllamaLLM:
    """Async Ollama LLM with proper cancellation support"""
    
    def __init__(
        self, 
        model: str = FIXED_MODEL, 
        base_url: str = OLLAMA_BASE_URL, 
        temperature: float = 0.7
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Async stream with proper cleanup"""
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"temperature": self.temperature}
            }
            
            async with self.client.stream('POST', url, json=payload, timeout=180.0) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if content := data.get("message", {}).get("content", ""):
                                    yield content
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    raise ValueError(f"Model error: {response.status_code}")
                    
        except httpx.ReadTimeout:
            raise ValueError("Request timed out")
        except Exception as e:
            raise ValueError(f"Streaming failed: {str(e)}")
    
    async def ainvoke(self, prompt: str) -> str:
        """Async invoke for non-streaming"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        
        response = await self.client.post(url, json=payload, timeout=180.0)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    
    async def close(self):
        """Close client"""
        await self.client.aclose()
