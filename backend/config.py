"""
Configuration management and constants
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
UPLOAD_DIR = Path("uploaded_documents")
VECTOR_DIR = Path("vector_data")
METADATA_FILE = VECTOR_DIR / "metadata.json"
CONFIG_FILE = VECTOR_DIR / "config.json"
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class ConfigManager:
    """Centralized configuration management"""
    
    DEFAULT_CONFIG = {
        'model': 'phi3',
        'embedding_model': 'nomic-embed-text',
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'temperature': 0.7,
        'timeout': 300,
        'cold_start_timeout': 900,
        'heartbeat_interval': 10,
        'heartbeat_enabled': True,
        'total_queries': 0
    }
    
    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self) -> None:
        """Load configuration from file with automatic upgrades"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Migrate old timeout values to new minimum thresholds
                if 'timeout' in loaded_config and loaded_config['timeout'] < 300:
                    logger.info(f"Upgrading timeout from {loaded_config['timeout']}s to 300s")
                    loaded_config['timeout'] = 300
                
                if 'cold_start_timeout' in loaded_config and loaded_config['cold_start_timeout'] < 900:
                    logger.info(f"Upgrading cold_start_timeout from {loaded_config['cold_start_timeout']}s to 900s")
                    loaded_config['cold_start_timeout'] = 900
                
                self.config.update(loaded_config)
                
                # Save upgraded config
                if any(key in loaded_config for key in ['timeout', 'cold_start_timeout']):
                    self.save()
            else:
                self.save()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save(self) -> None:
        """Save configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update(self, **kwargs) -> List[str]:
        """Update configuration and return list of changed fields"""
        changed = []
        for key, value in kwargs.items():
            if value is not None and key in self.config and self.config[key] != value:
                self.config[key] = value
                changed.append(key)
        
        if changed:
            self.save()
        
        return changed
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def increment_queries(self) -> None:
        """Increment query counter"""
        self.config['total_queries'] += 1
        self.save()
