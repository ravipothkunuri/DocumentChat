"""Document Processor - 40 lines"""
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from backend.config_manager import ConfigManager

class DocumentProcessor:
    """Handle document loading and processing"""
    
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader
    }
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def load_document(self, file_path: Path) -> List[str]:
        """Load document content based on file type"""
        suffix = file_path.suffix.lower()
        if suffix not in self.LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        loader = self.LOADERS[suffix](str(file_path))
        documents = loader.load()
        return [doc.page_content for doc in documents]
    
    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        """Split text content into chunks"""
        full_text = "\n\n".join(content)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size'),
            chunk_overlap=self.config.get('chunk_overlap'),
            length_function=len
        )
        
        chunks = splitter.split_text(full_text)
        
        return [
            {
                "page_content": chunk,
                "metadata": {"source": filename, "chunk_id": i}
            }
            for i, chunk in enumerate(chunks)
        ]
