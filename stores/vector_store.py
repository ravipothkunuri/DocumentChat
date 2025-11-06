"""In-memory vector store with cosine similarity and JSON persistence.

Holds document chunk metadata and embeddings, supports add/remove/clear,
top-k similarity search, and save/load to disk.
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from utils.errors import handle_errors


class VectorStore:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dimensions: Optional[int] = None
        self.last_update: Optional[str] = None

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not documents or not embeddings or len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must be non-empty and equal length")

        embedding_array = np.array(embeddings, dtype=np.float32)
        if self.embedding_dimensions is None:
            self.embedding_dimensions = embedding_array.shape[1]
        elif embedding_array.shape[1] != self.embedding_dimensions:
            raise ValueError("Embedding dimension mismatch")

        self.documents.extend(documents)
        self.embeddings = embedding_array if self.embeddings is None else np.vstack([self.embeddings, embedding_array])
        self.last_update = datetime.now().isoformat()

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_array = np.array([query_embedding], dtype=np.float32)
        query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)
        query_normalized = query_array / np.where(query_norm == 0, 1, query_norm)

        embeddings_norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        embeddings_normalized = self.embeddings / np.where(embeddings_norm == 0, 1, embeddings_norm)

        similarities = np.dot(query_normalized, embeddings_normalized.T)[0]
        top_k = min(k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.documents[idx], float(similarities[idx])) for idx in top_indices]

    def remove_documents_by_source(self, source: str) -> None:
        indices = [i for i, doc in enumerate(self.documents) if doc.get('metadata', {}).get('source') == source]
        if not indices:
            return

        for i in reversed(indices):
            del self.documents[i]

        if self.embeddings is not None:
            mask = np.ones(self.embeddings.shape[0], dtype=bool)
            mask[indices] = False
            self.embeddings = self.embeddings[mask] if mask.any() else None

        if self.embeddings is None:
            self.embedding_dimensions = None
        self.last_update = datetime.now().isoformat()

    def clear(self) -> None:
        self.documents = []
        self.embeddings = None
        self.embedding_dimensions = None
        self.last_update = datetime.now().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        return {"total_chunks": len(self.documents), "last_update": self.last_update}

    @handle_errors("Vector store save")
    def save(self) -> None:
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "embedding_dimensions": self.embedding_dimensions,
            "last_update": self.last_update,
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    @handle_errors("Vector store load")
    def load(self) -> None:
        from pathlib import Path

        if not Path(self.storage_path).exists():
            return
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        self.documents = data.get("documents", [])
        self.embedding_dimensions = data.get("embedding_dimensions")
        self.last_update = data.get("last_update")
        if embeddings_data := data.get("embeddings"):
            self.embeddings = np.array(embeddings_data, dtype=np.float32)


