"""File utilities for validation and constraints.

Contains `validate_file` used by the upload endpoint to enforce size and type.
"""
from pathlib import Path
from fastapi import HTTPException

from configuration import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB


def validate_file(filename: str, file_size: int) -> None:
    file_path = Path(filename)
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {file_path.suffix}")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(400, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")


