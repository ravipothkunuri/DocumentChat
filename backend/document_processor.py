"""
Document processing and text splitting
"""
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)

from backend.config import ConfigManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handle document loading and processing"""
    
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': UnstructuredWordDocumentLoader
    }
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def load_document(self, file_path: Path) -> List[str]:
        """Load document content based on file type"""
        suffix = file_path.suffix.lower()
        
        if suffix not in self.LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        try:
            if suffix == '.txt':
                loader = self.LOADERS[suffix](str(file_path), encoding='utf-8')
            else:
                loader = self.LOADERS[suffix](str(file_path))
            
            documents = loader.load()
            content = [doc.page_content for doc in documents]
            
            if not content or all(not c.strip() for c in content):
                raise ValueError("Document appears to be empty or contains no readable text")
            
            return content
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {file_path}, trying with latin-1")
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content_text = f.read()
                return [content_text]
            except Exception as e:
                raise ValueError(f"Failed to read file with multiple encodings: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load document: {str(e)}")
    
    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        """Split text content into chunks"""
        full_text = "\n\n".join(content)
        
        full_text = full_text.strip()
        if not full_text:
            raise ValueError("No text content to process after cleaning")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size'),
            chunk_overlap=self.config.get('chunk_overlap'),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(full_text)
        
        if not chunks:
            raise ValueError("Text splitting produced no chunks")
        
        documents = [
            {
                "page_content": chunk.strip(),
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "chunk_length": len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]
        
        if not documents:
            raise ValueError("No valid chunks after processing")
        
        logger.info(f"Split {filename} into {len(documents)} chunks")
        return documents
