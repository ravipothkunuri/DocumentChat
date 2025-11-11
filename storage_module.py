"""
Storage Module - Persistent Storage for DocumentChat

This module provides storage abstractions for the DocumentChat system:
- JSONStore: Simple key-value storage with JSON persistence
- VectorStore: Vector embeddings storage with similarity search
- DocumentStore: Unified facade combining both stores

The DocumentStore provides a high-level API that coordinates between
metadata storage and vector operations, simplifying application code.

Classes:
    JSONStore: Generic key-value persistence
    VectorStore: Vector similarity search engine
    DocumentStore: Unified document storage facade

Example:
    store = DocumentStore(Path("data"))
    store.add_document("doc.pdf", chunks, embeddings, 1024)
    results = store.search(query_embedding, k=5)
    docs = store.list_documents()

Author: Your Name
Version: 1.0.0
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Error Handling Decorator
# =============================================================================

def handle_errors(operation_name: str):
    """
    Decorator for consistent error handling and logging.
    
    Args:
        operation_name: Description of the operation for logging
        
    Returns:
        Decorated function that logs errors before raising
        
    Example:
        @handle_errors("Database operation")
        def save_data():
            # operation code
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}")
                raise
        return wrapper
    return decorator


# =============================================================================
# JSONStore - Simple Key-Value Storage
# =============================================================================

class JSONStore:
    """
    Simple JSON-based persistent key-value store.
    
    Provides thread-safe operations for storing and retrieving configuration
    and metadata. Data is automatically saved to disk on modifications.
    
    Attributes:
        filepath: Path to the JSON file
        data: In-memory dictionary of stored data
        
    Example:
        store = JSONStore(Path("config.json"), defaults={"count": 0})
        store.set("name", "DocumentChat")
        store.increment("count")
        value = store.get("name")  # Returns "DocumentChat"
    """
    
    def __init__(self, filepath: Path, defaults: Optional[Dict] = None):
        """
        Initialize the JSON store.
        
        Args:
            filepath: Path where JSON data will be stored
            defaults: Optional dictionary of default values
        """
        self.filepath = filepath
        self.data = defaults.copy() if defaults else {}
        self.load()

    @handle_errors("JSON load")
    def load(self) -> None:
        """
        Load data from JSON file if it exists.
        
        Merges loaded data with defaults. If file doesn't exist,
        uses only default values.
        """
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.data.update(json.load(f))

    @handle_errors("JSON save")
    def save(self) -> None:
        """
        Persist current data to JSON file.
        
        Writes formatted JSON with 2-space indentation.
        """
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str, default=None) -> Any:
        """
        Retrieve a value by key.
        
        Args:
            key: The key to look up
            default: Value to return if key doesn't exist
            
        Returns:
            The stored value or default
        """
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Store a value and persist to disk.
        
        Args:
            key: The key to store under
            value: The value to store (must be JSON-serializable)
        """
        self.data[key] = value
        self.save()

    def increment(self, key: str, amount: int = 1) -> None:
        """
        Increment a numeric value atomically.
        
        Args:
            key: The key to increment
            amount: Amount to add (default: 1)
            
        Example:
            store.increment("query_count")  # Increments by 1
            store.increment("errors", 5)    # Increments by 5
        """
        self.data[key] = self.data.get(key, 0) + amount
        self.save()

    def remove(self, key: str) -> bool:
        """
        Remove a key-value pair.
        
        Args:
            key: The key to remove
            
        Returns:
            True if key existed and was removed, False otherwise
        """
        if key in self.data:
            del self.data[key]
            self.save()
            return True
        return False

    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.data

    def all_values(self) -> List[Any]:
        """
        Get all stored values.
        
        Returns:
            List of all values in the store
        """
        return list(self.data.values())
    
    def clear(self) -> None:
        """
        Clear all data from the store.
        
        Resets the store to empty state and persists.
        """
        self.data = {}
        self.save()


# =============================================================================
# VectorStore - Vector Embeddings with Similarity Search
# =============================================================================

class VectorStore:
    """
    Custom vector store for document embeddings and similarity search.
    
    Uses numpy for efficient cosine similarity calculations. Stores document
    chunks with their embeddings and provides similarity search functionality.
    
    Attributes:
        storage_path: Path to persistence file
        documents: List of document chunks with metadata
        embeddings: Numpy array of embedding vectors
        embedding_dimensions: Dimension of embedding vectors
        last_update: ISO timestamp of last update
        
    Example:
        store = VectorStore()
        store.add_documents(docs, embeddings)
        results = store.similarity_search(query_embedding, k=5)
        for doc, score in results:
            print(f"Score: {score}, Content: {doc['page_content']}")
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize the vector store.
        
        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dimensions: Optional[int] = None
        self.last_update: Optional[str] = None

    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ) -> None:
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of document dictionaries with 'page_content' and 'metadata'
            embeddings: List of embedding vectors (same length as documents)
            
        Raises:
            ValueError: If documents/embeddings are empty, lengths don't match,
                       or embedding dimensions are inconsistent
                       
        Example:
            docs = [
                {"page_content": "text", "metadata": {"source": "file.pdf"}},
                {"page_content": "more text", "metadata": {"source": "file.pdf"}}
            ]
            embeds = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            store.add_documents(docs, embeds)
        """
        # Validate inputs
        if not documents or not embeddings or len(documents) != len(embeddings):
            raise ValueError(
                "Documents and embeddings must be non-empty and equal length"
            )

        # Convert to numpy array for efficient operations
        embedding_array = np.array(embeddings, dtype=np.float32)
        
        # Check/set embedding dimensions
        if self.embedding_dimensions is None:
            self.embedding_dimensions = embedding_array.shape[1]
        elif embedding_array.shape[1] != self.embedding_dimensions:
            raise ValueError(f"Embedding dimension mismatch")

        # Add documents and embeddings
        self.documents.extend(documents)
        
        # Stack embeddings vertically
        if self.embeddings is None:
            self.embeddings = embedding_array
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_array])
        
        self.last_update = datetime.now().isoformat()

    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 4
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find k most similar documents using cosine similarity.
        
        Args:
            query_embedding: Embedding vector of the query
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples, sorted by score descending
            
        Note:
            Similarity scores are cosine similarities in range [-1, 1],
            where 1 indicates identical vectors
            
        Example:
            query_emb = embeddings_model.embed_query("What is AI?")
            results = store.similarity_search(query_emb, k=3)
            for doc, score in results:
                print(f"Similarity: {score:.3f}")
                print(f"Content: {doc['page_content'][:100]}...")
        """
        # Return empty if no documents
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)
        query_normalized = query_array / np.where(query_norm == 0, 1, query_norm)

        # Normalize stored embeddings
        embeddings_norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        embeddings_normalized = self.embeddings / np.where(
            embeddings_norm == 0, 1, embeddings_norm
        )

        # Calculate cosine similarities
        similarities = np.dot(query_normalized, embeddings_normalized.T)[0]
        
        # Get top k results
        top_k = min(k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            (self.documents[idx], float(similarities[idx])) 
            for idx in top_indices
        ]

    def remove_documents_by_source(self, source: str) -> None:
        """
        Remove all documents from a specific source file.
        
        Args:
            source: Source filename to remove
            
        Example:
            store.remove_documents_by_source("document.pdf")
            # Removes all chunks from document.pdf
        """
        # Find indices of documents from this source
        indices = [
            i for i, doc in enumerate(self.documents) 
            if doc.get('metadata', {}).get('source') == source
        ]
        
        if not indices:
            return

        # Remove documents in reverse order to maintain indices
        for i in reversed(indices):
            del self.documents[i]

        # Remove corresponding embeddings using boolean mask
        if self.embeddings is not None:
            mask = np.ones(self.embeddings.shape[0], dtype=bool)
            mask[indices] = False
            self.embeddings = self.embeddings[mask] if mask.any() else None

        # Reset dimensions if no embeddings left
        if self.embeddings is None:
            self.embedding_dimensions = None
            
        self.last_update = datetime.now().isoformat()

    def clear(self) -> None:
        """
        Clear all documents and embeddings from the store.
        
        Resets the store to initial empty state.
        """
        self.documents = []
        self.embeddings = None
        self.embedding_dimensions = None
        self.last_update = datetime.now().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with total_chunks and last_update
        """
        return {
            "total_chunks": len(self.documents),
            "last_update": self.last_update
        }

    @handle_errors("Vector store save")
    def save(self) -> None:
        """
        Persist vector store to JSON file.
        
        Converts numpy arrays to lists for JSON serialization.
        """
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "embedding_dimensions": self.embedding_dimensions,
            "last_update": self.last_update
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    @handle_errors("Vector store load")
    def load(self) -> None:
        """
        Load vector store from JSON file.
        
        Converts embedding lists back to numpy arrays.
        """
        if not self.storage_path.exists():
            return
            
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
            
        self.documents = data.get("documents", [])
        self.embedding_dimensions = data.get("embedding_dimensions")
        self.last_update = data.get("last_update")
        
        # Convert embeddings back to numpy array
        if embeddings_data := data.get("embeddings"):
            self.embeddings = np.array(embeddings_data, dtype=np.float32)
            
        logger.info(f"Loaded {len(self.documents)} documents from vector store")


# =============================================================================
# DocumentStore - Unified Storage Facade
# =============================================================================

class DocumentStore:
    """
    Unified document storage facade combining metadata and vector storage.
    
    Provides a high-level API for document operations while delegating to
    specialized storage backends. Coordinates between JSONStore for metadata
    and VectorStore for embeddings.
    
    This facade simplifies application code by providing atomic operations
    that handle both metadata and vector storage automatically.
    
    Attributes:
        config: Configuration storage (JSONStore)
        metadata: Document metadata storage (JSONStore)
        vectors: Vector embeddings storage (VectorStore)
        
    Example:
        store = DocumentStore(Path("data"))
        
        # Add document (handles both metadata and vectors)
        store.add_document("doc.pdf", chunks, embeddings, 1024)
        
        # Search (uses vector store)
        results = store.search(query_embedding, k=5)
        
        # List documents (uses metadata store)
        docs = store.list_documents()
        
        # Get statistics (aggregates from both stores)
        stats = store.get_stats()
    """
    
    def __init__(self, base_path: Path):
        """
        Initialize the document store.
        
        Creates three internal stores:
        - config: Application configuration
        - metadata: Document metadata
        - vectors: Document embeddings
        
        Args:
            base_path: Base directory for all storage files
        """
        base_path.mkdir(exist_ok=True)
        
        # Initialize component stores
        self.config = JSONStore(
            base_path / "config.json",
            defaults={
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'temperature': 0.7,
                'total_queries': 0
            }
        )
        
        self.metadata = JSONStore(base_path / "metadata.json")
        self.vectors = VectorStore(base_path / "vectors.json")
        
        # Load existing vector data
        try:
            self.vectors.load()
            logger.info("DocumentStore initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
    
    def add_document(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        file_size: int,
        file_type: str
    ) -> None:
        """
        Add a document with both metadata and vector embeddings.
        
        This is an atomic operation that stores both the document metadata
        and its vector embeddings, then persists both stores.
        
        Args:
            filename: Name of the document file
            chunks: List of document chunks with content and metadata
            embeddings: List of embedding vectors for each chunk
            file_size: Size of the original file in bytes
            file_type: File extension (pdf, txt, docx)
            
        Example:
            chunks = [
                {"page_content": "AI is...", "metadata": {"source": "doc.pdf", "chunk_id": 0}},
                {"page_content": "Machine learning...", "metadata": {"source": "doc.pdf", "chunk_id": 1}}
            ]
            embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            store.add_document("doc.pdf", chunks, embeddings, 1024000, "pdf")
        """
        # Store vector embeddings
        self.vectors.add_documents(chunks, embeddings)
        
        # Store metadata
        self.metadata.set(filename, {
            "filename": filename,
            "size": file_size,
            "chunks": len(chunks),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_type
        })
        
        # Persist both stores
        self.save_all()
    
    def remove_document(self, filename: str) -> None:
        """
        Remove a document from both metadata and vector stores.
        
        Args:
            filename: Name of the document to remove
            
        Example:
            store.remove_document("old_report.pdf")
        """
        # Remove from vector store
        self.vectors.remove_documents_by_source(filename)
        
        # Remove metadata
        self.metadata.remove(filename)
        
        # Persist changes
        self.save_all()
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 4
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Embedding vector of the search query
            k: Number of results to return
            
        Returns:
            List of (document_chunk, similarity_score) tuples
            
        Example:
            query_emb = embeddings_model.embed_query("What is AI?")
            results = store.search(query_emb, k=5)
            
            for doc, score in results:
                print(f"Source: {doc['metadata']['source']}")
                print(f"Score: {score:.3f}")
                print(f"Content: {doc['page_content'][:100]}...")
        """
        return self.vectors.similarity_search(query_embedding, k)
    
    def get_document_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document.
        
        Args:
            filename: Name of the document
            
        Returns:
            Metadata dictionary or None if not found
            
        Example:
            info = store.get_document_info("report.pdf")
            if info:
                print(f"Chunks: {info['chunks']}, Size: {info['size']}")
        """
        return self.metadata.get(filename)
    
    def document_exists(self, filename: str) -> bool:
        """
        Check if a document exists in the store.
        
        Args:
            filename: Name of the document
            
        Returns:
            True if document exists, False otherwise
        """
        return self.metadata.exists(filename)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Get a list of all documents with their metadata.
        
        Returns:
            List of metadata dictionaries
            
        Example:
            docs = store.list_documents()
            for doc in docs:
                print(f"{doc['filename']}: {doc['chunks']} chunks")
        """
        return self.metadata.all_values()
    
    def clear_all(self) -> None:
        """
        Clear all documents from both metadata and vector stores.
        
        This removes all stored data but preserves configuration.
        
        Example:
            store.clear_all()  # Removes all documents
        """
        self.vectors.clear()
        self.metadata.clear()
        self.save_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all stores.
        
        Returns:
            Dictionary with aggregated statistics
            
        Example:
            stats = store.get_stats()
            print(f"Documents: {stats['total_documents']}")
            print(f"Chunks: {stats['total_chunks']}")
            print(f"Queries: {stats['total_queries']}")
        """
        vector_stats = self.vectors.get_stats()
        doc_count = len(self.metadata.data)
        
        # Calculate total storage size
        total_size = sum(
            meta.get('size', 0) 
            for meta in self.metadata.data.values()
        )
        
        return {
            "total_documents": doc_count,
            "total_chunks": vector_stats.get('total_chunks', 0),
            "total_queries": self.config.get('total_queries', 0),
            "total_storage_size": total_size,
            "average_chunks_per_document": round(
                vector_stats.get('total_chunks', 0) / max(1, doc_count), 2
            ),
            "last_update": vector_stats.get('last_update')
        }
    
    def increment_query_count(self) -> None:
        """
        Increment the total query counter.
        
        Called after each successful query operation.
        """
        self.config.increment('total_queries')
    
    def get_config(self, key: str, default=None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to store
        """
        self.config.set(key, value)
    
    def save_all(self) -> None:
        """
        Persist all stores to disk.
        
        Saves configuration, metadata, and vector data.
        """
        self.config.save()
        self.metadata.save()
        self.vectors.save()
