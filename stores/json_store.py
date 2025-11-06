"""Lightweight JSON-backed key-value store with basic helpers.

Provides get/set/increment/remove/exists and lazy load/save behavior.
"""
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

from utils.errors import handle_errors


class JSONStore:
    def __init__(self, filepath: Path, defaults: Optional[Dict] = None):
        self.filepath = filepath
        self.data = defaults.copy() if defaults else {}
        self.load()

    @handle_errors("JSON load")
    def load(self) -> None:
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.data.update(json.load(f))

    @handle_errors("JSON save")
    def save(self) -> None:
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str, default=None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def increment(self, key: str, amount: int = 1) -> None:
        self.data[key] = self.data.get(key, 0) + amount
        self.save()

    def remove(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            self.save()
            return True
        return False

    def exists(self, key: str) -> bool:
        return key in self.data

    def all_values(self) -> List[Any]:
        return list(self.data.values())


