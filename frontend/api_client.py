"""API client with async streaming"""
import httpx
import json
from typing import List, Dict, Tuple, Optional, AsyncIterator

class APIClient:
    """Async API client for backend interactions"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.sync_client = httpx.Client(timeout=60.0)
        self.async_client = httpx.AsyncClient(timeout=60.0)
    
    def _request(self, method: str, endpoint: str, timeout: int = 10, **kwargs) -> Tuple[int, Dict]:
        """Synchronous request handler"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.sync_client.request(method, url, timeout=timeout, **kwargs)
            data = response.json() if response.content else {}
            return response.status_code, data
        except json.JSONDecodeError:
            return response.status_code, {"message": "Invalid response"}
        except httpx.RequestError as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def health_check(self) -> Tuple[bool, Optional[Dict]]:
        """Check backend health"""
        try:
            response = self.sync_client.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.is_success else None
        except httpx.RequestError:
            return False, None
    
    def get_documents(self) -> List[Dict]:
        """Get list of documents"""
        status_code, data = self._request('GET', '/documents')
        return data if status_code == 200 else []
    
    def upload_file(self, file) -> Tuple[int, Dict]:
        """Upload a file"""
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.sync_client.post(f"{self.base_url}/upload", files=files, timeout=60)
            return response.status_code, response.json() if response.content else {}
        except Exception as e:
            return 500, {"message": f"Upload error: {str(e)}"}
    
    def delete_document(self, filename: str) -> Tuple[int, Dict]:
        """Delete a document"""
        return self._request('DELETE', f'/documents/{filename}', timeout=30)
    
    async def query_stream(self, question: str, top_k: int = 4) -> AsyncIterator[Dict]:
        """Async stream query response with proper cancellation support"""
        try:
            payload = {
                "question": question,
                "stream": True,
                "top_k": top_k,
                "model": "llama3.2"
            }
            
            async with self.async_client.stream(
                'POST',
                f"{self.base_url}/query",
                json=payload,
                timeout=120.0
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line and line.startswith('data: '):
                            try:
                                yield json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                else:
                    yield {"type": "error", "message": f"Query failed: {response.status_code}"}
                    
        except httpx.ReadTimeout:
            yield {"type": "error", "message": "Request timed out"}
        except httpx.RemoteProtocolError:
            yield {"type": "error", "message": "Connection interrupted"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
    
    def __del__(self):
        """Cleanup clients"""
        try:
            self.sync_client.close()
        except:
            pass
