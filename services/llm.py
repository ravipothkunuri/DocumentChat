"""Asynchronous client wrapper for the Ollama chat API used for streaming.

Provides `AsyncOllamaLLM` with an `astream` generator yielding model tokens
and a `close` method to release the underlying HTTP client.
"""
import json
from typing import AsyncIterator

import httpx


class AsyncOllamaLLM:
    def __init__(self, model: str, base_url: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"temperature": self.temperature},
            }
            async with self.client.stream('POST', url, json=payload, timeout=180.0) as response:
                if response.status_code != 200:
                    raise ValueError(f"Ollama returned HTTP {response.status_code}")
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
        except httpx.ReadTimeout:
            raise ValueError("Request timed out")
        except httpx.ConnectError:
            raise ValueError("Cannot connect to Ollama service")
        except Exception as e:
            raise ValueError(f"Stream error: {str(e)}")

    async def close(self) -> None:
        await self.client.aclose()


