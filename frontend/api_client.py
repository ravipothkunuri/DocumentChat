"""HTTP client used by the Streamlit UI to talk to the backend API."""
import json
from typing import Dict, Optional, Tuple, List, AsyncIterator

import httpx


class APIClient:
    def __init__(self, base_url: str, timeout_seconds: float = 60.0):
        self.base_url = base_url
        self.sync_client = httpx.Client(timeout=timeout_seconds)
        self.async_client = httpx.AsyncClient(timeout=timeout_seconds)

    def _make_request(self, method: str, endpoint: str, timeout: int = 10, **kwargs) -> Tuple[int, Dict]:
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.sync_client.request(method, url, timeout=timeout, **kwargs)
            data = response.json() if response.content else {}
            return response.status_code, data
        except json.JSONDecodeError:
            return response.status_code, {"message": "Invalid JSON response"}
        except httpx.RequestError as e:
            return 500, {"message": f"Connection error: {str(e)}"}

    def health_check(self) -> Tuple[bool, Optional[Dict]]:
        try:
            response = self.sync_client.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.is_success else None
        except httpx.RequestError:
            return False, None

    def get_documents(self) -> List[Dict]:
        status_code, data = self._make_request('GET', '/documents')
        return data if status_code == 200 else []

    def upload_file(self, file) -> Tuple[int, Dict]:
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.sync_client.post(f"{self.base_url}/upload", files=files, timeout=60)
            return response.status_code, response.json() if response.content else {}
        except Exception as e:
            return 500, {"message": f"Upload failed: {str(e)}"}

    def delete_document(self, filename: str) -> Tuple[int, Dict]:
        return self._make_request('DELETE', f'/documents/{filename}', timeout=30)

    async def query_stream(self, question: str, model: str, top_k: int = 4) -> AsyncIterator[Dict]:
        try:
            payload = {"question": question, "stream": True, "top_k": top_k, "model": model}
            async with self.async_client.stream('POST', f"{self.base_url}/query", json=payload, timeout=120.0) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line and line.startswith('data: '):
                            try:
                                yield json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                else:
                    yield {"type": "error", "message": f"Query failed with status {response.status_code}"}
        except httpx.ReadTimeout:
            yield {"type": "error", "message": "Request timed out"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    def __del__(self):
        try:
            self.sync_client.close()
        except Exception:
            pass


