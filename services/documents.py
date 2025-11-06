"""Document ingestion and splitting utilities for supported file types.

`DocumentProcessor` loads content using LangChain loaders and splits text
into retrievable chunks with configurable size and overlap.
"""
from typing import List, Dict, Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader

from stores.json_store import JSONStore


class DocumentProcessor:
    LOADERS = {'.pdf': PyMuPDFLoader, '.txt': TextLoader, '.docx': Docx2txtLoader}

    def __init__(self, config: JSONStore):
        self.config = config

    def load_document(self, file_path: Path) -> List[str]:
        suffix = file_path.suffix.lower()
        if suffix not in self.LOADERS:
            raise ValueError(f"Unsupported file type: {suffix}")
        loader = self.LOADERS[suffix](str(file_path))
        documents = loader.load()
        return [doc.page_content for doc in documents]

    def split_text(self, content: List[str], filename: str) -> List[Dict[str, Any]]:
        full_text = "\n\n".join(content)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size', 1000),
            chunk_overlap=self.config.get('chunk_overlap', 200),
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(full_text)
        return [
            {"page_content": chunk, "metadata": {"source": filename, "chunk_id": i}}
            for i, chunk in enumerate(chunks)
        ]


