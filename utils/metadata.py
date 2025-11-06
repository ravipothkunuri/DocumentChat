"""Metadata helpers for constructing canonical upload metadata objects."""
from datetime import datetime


def build_upload_metadata(filename: str, size: int, chunks: int, suffix: str) -> dict:
    return {
        "filename": filename,
        "size": size,
        "chunks": chunks,
        "status": "processed",
        "uploaded_at": datetime.now().isoformat(),
        "type": suffix[1:].lower(),
    }


