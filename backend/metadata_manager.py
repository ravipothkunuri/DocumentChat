"""Metadata Manager - 50 lines"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from backend.config import METADATA_FILE

logger = logging.getLogger(__name__)

class MetadataManager:
    """Manage document metadata"""
    
    def __init__(self, metadata_file: Path = METADATA_FILE):
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self):
        """Load metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}
    
    def save(self):
        """Save metadata to file"""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add(self, filename: str, metadata: Dict[str, Any]):
        """Add or update document metadata"""
        self.metadata[filename] = metadata
        self.save()
    
    def remove(self, filename: str) -> bool:
        """Remove document metadata"""
        if filename in self.metadata:
            del self.metadata[filename]
            self.save()
            return True
        return False
    
    def get(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        return self.metadata.get(filename)
    
    def exists(self, filename: str) -> bool:
        """Check if document exists"""
        return filename in self.metadata
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all document metadata"""
        return list(self.metadata.values())
