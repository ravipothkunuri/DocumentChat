"""HTTP utility helpers for consistent responses.

Currently exposes `http_response` to standardize success/error payloads.
"""
from typing import Dict, Any


def http_response(status_code: int, message: str, **extra_data) -> Dict[str, Any]:
    return {"status": "success" if status_code < 400 else "error", "message": message, **extra_data}


