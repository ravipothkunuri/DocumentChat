import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Custom in-memory vector store with JSON persistence."""
    
    def __init__(self, storage_path: str = "vector_data/vectors.json"):
        self.storage_path = Path(storage_path)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dimensions: Optional[int] = None
        self.last_update: Optional[str] = None
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(exist_ok=True)
        logger.debug(f"VectorStore initialized with storage: {self.storage_path}")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the store."""
        if not documents or not embeddings:
            raise ValueError("Documents and embeddings cannot be empty")
        
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents, {len(embeddings)} embeddings")
        
        # Convert to numpy array
        embedding_array = np.array(embeddings, dtype=np.float32)
        
        # Initialize or validate dimensions
        if self.embedding_dimensions is None:
            self.embedding_dimensions = embedding_array.shape[1]
            logger.info(f"Set embedding dimensions to {self.embedding_dimensions}")
        elif embedding_array.shape[1] != self.embedding_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimensions}, "
                f"got {embedding_array.shape[1]}"
            )
        
        # Add documents
        self.documents.extend(documents)
        
        # Add embeddings
        self.embeddings = (
            embedding_array if self.embeddings is None
            else np.vstack([self.embeddings, embedding_array])
        )
        
        self.last_update = datetime.now().isoformat()
        
        logger.debug(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 4
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find the most similar documents to the query."""
        if self.embeddings is None or len(self.documents) == 0:
            logger.warning("No documents in vector store for similarity search")
            return []
        
        if len(query_embedding) != self.embedding_dimensions:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimensions}, "
                f"got {len(query_embedding)}"
            )
        
        # Convert query to numpy array and calculate cosine similarity
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Normalize vectors for cosine similarity with zero-vector protection
        query_norms = np.linalg.norm(query_array, axis=1, keepdims=True)
        query_norms = np.where(query_norms == 0, 1, query_norms)  # Avoid division by zero
        query_norm = query_array / query_norms
        
        embeddings_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        embeddings_norms = np.where(embeddings_norms == 0, 1, embeddings_norms)  # Avoid division by zero
        embeddings_norm = self.embeddings / embeddings_norms
        
        # Compute cosine similarity via dot product of normalized vectors
        similarities = np.dot(query_norm, embeddings_norm.T)[0]
        
        # Get top k indices
        top_k = min(k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = [(self.documents[idx], float(similarities[idx])) for idx in top_indices]
        
        logger.debug(f"Similarity search returned {len(results)} results")
        return results
    
    def remove_documents_by_source(self, source: str) -> None:
        """Remove all documents from a specific source."""
        indices_to_remove = [
            i for i, doc in enumerate(self.documents)
            if doc.get('metadata', {}).get('source') == source
        ]
        
        if not indices_to_remove:
            logger.warning(f"No documents found for source: {source}")
            return
        
        # Remove documents in reverse order
        for i in reversed(indices_to_remove):
            del self.documents[i]
        
        # Remove corresponding embeddings
        if self.embeddings is not None:
            mask = np.ones(self.embeddings.shape[0], dtype=bool)
            mask[indices_to_remove] = False
            self.embeddings = self.embeddings[mask] if mask.any() else None
            
            if self.embeddings is None:
                self.embedding_dimensions = None
        
        self.last_update = datetime.now().isoformat()
        logger.info(f"Removed {len(indices_to_remove)} documents from source: {source}")
    
    def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None
        self.embedding_dimensions = None
        self.last_update = datetime.now().isoformat()
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": len(self.documents),
            "embedding_dimensions": self.embedding_dimensions,
            "last_update": self.last_update,
            "dimension_consistent": True,
            "vector_store_size": 0
        }
        
        if self.embeddings is not None:
            stats["dimension_consistent"] = self.embeddings.shape[1] == self.embedding_dimensions
            stats["vector_store_size"] = self.embeddings.nbytes
        
        return stats
    
    def save(self) -> None:
        """Save the vector store to disk."""
        try:
            data = {
                "documents": self.documents,
                "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
                "embedding_dimensions": self.embedding_dimensions,
                "last_update": self.last_update,
                "version": "1.0"
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Vector store saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load(self) -> None:
        """Load the vector store from disk."""
        if not self.storage_path.exists():
            logger.info("No existing vector store file found")
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.documents = data.get("documents", [])
            self.embedding_dimensions = data.get("embedding_dimensions")
            self.last_update = data.get("last_update")
            
            # Load embeddings
            embeddings_data = data.get("embeddings")
            if embeddings_data:
                self.embeddings = np.array(embeddings_data, dtype=np.float32)
                
                # Validate consistency
                if len(self.documents) != self.embeddings.shape[0]:
                    raise ValueError(
                        f"Corrupted store: {len(self.documents)} documents, "
                        f"{self.embeddings.shape[0]} embeddings"
                    )
                
                if self.embeddings.shape[1] != self.embedding_dimensions:
                    raise ValueError(
                        f"Corrupted store: dimension mismatch "
                        f"(expected {self.embedding_dimensions}, got {self.embeddings.shape[1]})"
                    )
            else:
                self.embeddings = None
            
            logger.info(
                f"Vector store loaded: {len(self.documents)} documents, "
                f"dimensions: {self.embedding_dimensions}"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in vector store file: {e}")
            self._handle_corrupted_store("invalid JSON")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self._handle_corrupted_store(str(e))
    
    def _handle_corrupted_store(self, reason: str) -> None:
        """Handle corrupted vector store by clearing and saving."""
        logger.warning(f"Clearing corrupted vector store: {reason}")
        self.clear()
        self.save()
        raise ValueError(f"Corrupted vector store cleared: {reason}")
    
    def validate_consistency(self) -> bool:
        """Validate internal consistency of the vector store."""
        try:
            # Check document/embedding count
            if self.embeddings is not None:
                if len(self.documents) != self.embeddings.shape[0]:
                    logger.error(
                        f"Count mismatch: {len(self.documents)} docs, "
                        f"{self.embeddings.shape[0]} embeddings"
                    )
                    return False
                
                # Check dimensions
                if self.embeddings.shape[1] != self.embedding_dimensions:
                    logger.error(
                        f"Dimension mismatch: expected {self.embedding_dimensions}, "
                        f"got {self.embeddings.shape[1]}"
                    )
                    return False
            
            # Check document structure
            for i, doc in enumerate(self.documents):
                if 'metadata' not in doc or 'page_content' not in doc:
                    logger.error(f"Invalid document structure at index {i}")
                    return False
            
            logger.debug("Vector store consistency validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating consistency: {e}")
            return False
