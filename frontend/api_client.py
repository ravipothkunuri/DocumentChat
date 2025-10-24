"""
API client for backend interactions
"""
import requests
import json
from typing import List, Dict, Tuple, Optional
from config import STREAM_TIMEOUT


class RAGAPIClient:
    """Centralized API client for all backend interactions"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, timeout: int = 10, **kwargs) -> Tuple[int, Dict]:
        """Unified request handler with error handling."""
        response = None
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, timeout=timeout, **kwargs)
            data = response.json() if response.content else {}
            return response.status_code, data
        except json.JSONDecodeError:
            if response is not None:
                return response.status_code, {"message": "Invalid response from server"}
            return 500, {"message": "Invalid response from server"}
        except requests.exceptions.RequestException as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def health_check(self) -> Tuple[bool, Optional[Dict]]:
        """Check backend health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.ok else None
        except requests.exceptions.RequestException:
            return False, None
    
    def get_models(self) -> Dict:
        """Get available models"""
        from config import DEFAULT_MODELS
        status_code, data = self._request('GET', '/models')
        return data if status_code == 200 else DEFAULT_MODELS
    
    def get_documents(self) -> List[Dict]:
        """Get list of uploaded documents"""
        status_code, data = self._request('GET', '/documents')
        if status_code == 200 and isinstance(data, list):
            return data
        return []
    
    def upload_file(self, file) -> Tuple[int, Dict]:
        """Upload a file to the backend"""
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.session.post(f"{self.base_url}/upload", files=files, timeout=60)
            return response.status_code, response.json() if response.content else {}
        except Exception as e:
            return 500, {"message": f"Upload error: {str(e)}"}
    
    def delete_document(self, filename: str) -> Tuple[int, Dict]:
        """Delete a specific document"""
        return self._request('DELETE', f'/documents/{filename}', timeout=30)
    
    def get_document_details(self, filename: str) -> Tuple[int, Dict]:
        """Get detailed information about a specific document"""
        if not filename:
            return 400, {"message": "Filename cannot be empty"}
        return self._request('GET', f'/documents/{filename}/details', timeout=10)
    
    def query_stream(self, question: str, top_k: int = 4, model: Optional[str] = None):
        """Stream query response with proper connection cleanup"""
        response = None
        try:
            payload = {"question": question, "stream": True, "top_k": top_k}
            if model:
                payload["model"] = model
            
            # Use configured timeout to accommodate heartbeat mechanism
            # Backend sends heartbeat every 10s by default, so timeout won't be triggered
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                stream=True,
                timeout=STREAM_TIMEOUT
            )
            
            if response.status_code == 200:
                try:
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                yield json.loads(line_text[6:])
                finally:
                    # Always close the response to free up the connection
                    response.close()
            else:
                yield {"type": "error", "message": f"Query failed with status {response.status_code}"}
                response.close()
        except requests.exceptions.ChunkedEncodingError:
            yield {"type": "error", "message": "Connection interrupted"}
        except requests.exceptions.ConnectionError:
            yield {"type": "error", "message": "Lost connection to backend"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            # Ensure response is closed even if an exception occurred
            if response is not None:
                response.close()
