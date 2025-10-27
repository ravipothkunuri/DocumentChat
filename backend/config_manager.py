"""Configuration Manager - 45 lines"""
import json
import logging
from pathlib import Path
from backend.config import CONFIG_FILE, FIXED_MODEL, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management"""
    
    DEFAULT_CONFIG = {
        'model': FIXED_MODEL,
        'embedding_model': DEFAULT_EMBEDDING_MODEL,
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'temperature': 0.7,
        'total_queries': 0
    }
    
    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
            else:
                self.save()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save(self):
        """Save configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def increment_queries(self):
        """Increment query counter"""
        self.config['total_queries'] += 1
        self.save()
