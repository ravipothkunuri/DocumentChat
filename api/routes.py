"""FastAPI router and endpoint implementations for DocumentChat.

Initializes stores and services, exposes health, upload, query, documents
management, clear, and stats endpoints. This module is imported by the app
entrypoint to include the router and share the lifecycle-managed model manager.
"""
import json
import logging
from datetime import datetime
from typing import List, Tuple
from pathlib import Path

import httpx
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse

from configuration import (
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    UPLOAD_DIR,
    VECTOR_DIR,
)
from schemas.query import QueryRequest, DocumentUploadResponse, DocumentInfo
from services.documents import DocumentProcessor
from services.model_manager import ModelManager
from stores.json_store import JSONStore
from stores.vector_store import VectorStore
from utils.files import validate_file
from utils.http import http_response
from utils.metadata import build_upload_metadata


logger = logging.getLogger(__name__)


def check_ollama_health() -> Tuple[bool, str]:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200, "Available" if response.status_code == 200 else f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


# Initialize stores/services
config_store = JSONStore(
    VECTOR_DIR / "config.json",
    defaults={
        'model': DEFAULT_MODEL,
        'embedding_model': DEFAULT_EMBEDDING_MODEL,
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'temperature': 0.7,
        'total_queries': 0,
    },
)
metadata_store = JSONStore(VECTOR_DIR / "metadata.json")
vector_store = VectorStore(VECTOR_DIR / "vectors.json")
model_manager = ModelManager(config_store)
document_processor = DocumentProcessor(config_store)

try:
    vector_store.load()
    logger.info("Vector store loaded successfully")
except Exception as e:
    logger.warning(f"Could not load vector store: {e}")


router = APIRouter()


@router.get("/health")
async def health_check():
    ollama_available, ollama_message = check_ollama_health()
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ollama_status": {"available": ollama_available, "message": ollama_message},
        "configuration": {
            "model": DEFAULT_MODEL,
            "embedding_model": config_store.get('embedding_model'),
            "chunk_size": config_store.get('chunk_size'),
        },
        "document_count": len(metadata_store.data),
        "total_chunks": stats.get("total_chunks", 0),
        "total_queries": config_store.get('total_queries'),
    }


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    validate_file(file.filename, len(content))

    if metadata_store.exists(file.filename):
        raise HTTPException(400, f"Document '{file.filename}' already exists")

    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, 'wb') as f:
            f.write(content)

        doc_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(doc_content, file.filename)

        if not documents:
            raise ValueError("No content extracted from document")

        texts = [doc["page_content"] for doc in documents]
        embeddings = model_manager.embeddings_model.embed_documents(texts)
        vector_store.add_documents(documents, embeddings)

        metadata_store.set(
            file.filename,
            build_upload_metadata(file.filename, len(content), len(documents), file_path.suffix),
        )
        vector_store.save()
        logger.info(f"Successfully processed {file.filename}: {len(documents)} chunks")

        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=len(content),
            message=f"Processed into {len(documents)} chunks",
        )
    except Exception as e:
        logger.error(f"Processing failed for {file.filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        metadata_store.remove(file.filename)
        raise HTTPException(500, f"Processing failed: {str(e)}")


@router.post("/query")
async def query_documents(request: Request, query: QueryRequest):
    from utils.prompts import build_chat_prompt

    start_time = datetime.now()
    try:
        if vector_store.get_stats().get("total_chunks", 0) == 0:
            raise HTTPException(400, "No documents available for querying")

        query_embedding = model_manager.embeddings_model.embed_query(query.question)
        similar_docs = vector_store.similarity_search(query_embedding, k=query.top_k)

        if not similar_docs:
            raise HTTPException(400, "No relevant documents found")

        context_parts = []
        sources = []
        scores = []
        for doc, score in similar_docs:
            context_parts.append(f"Source: {doc['metadata']['source']}\nContent: {doc['page_content']}")
            sources.append(doc['metadata']['source'])
            scores.append(float(score))

        context = "\n\n".join(context_parts)
        unique_sources = list(set(sources))
        doc_identity = unique_sources[0] if len(unique_sources) == 1 else "your documents"

        prompt = build_chat_prompt(doc_identity, context, query.question)

        llm = model_manager.llm

        if query.stream:
            async def generate():
                try:
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": scores,
                        "model_used": DEFAULT_MODEL,
                        "type": "metadata",
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"

                    async for chunk in llm.astream(prompt):
                        if await request.is_disconnected():
                            break
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                    processing_time = (datetime.now() - start_time).total_seconds()
                    yield f"data: {json.dumps({'type': 'done', 'processing_time': processing_time})}\n\n"
                    config_store.increment('total_queries')
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            full_response = ""
            async for chunk in llm.astream(prompt):
                full_response += chunk

            processing_time = (datetime.now() - start_time).total_seconds()
            config_store.increment('total_queries')

            return {
                "answer": full_response,
                "sources": sources,
                "chunks_used": len(similar_docs),
                "similarity_scores": scores,
                "processing_time": processing_time,
                "model_used": DEFAULT_MODEL,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    return [DocumentInfo(**meta) for meta in metadata_store.all_values()]


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    if not metadata_store.exists(filename):
        raise HTTPException(404, f"Document '{filename}' not found")
    try:
        vector_store.remove_documents_by_source(filename)
        metadata_store.remove(filename)
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        vector_store.save()
        return http_response(200, f"Document '{filename}' deleted successfully")
    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")


@router.delete("/clear")
async def clear_all_documents():
    try:
        vector_store.clear()
        metadata_store.data = {}
        metadata_store.save()
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        vector_store.save()
        return http_response(200, "All documents cleared", cleared=True)
    except Exception as e:
        raise HTTPException(500, f"Clear operation failed: {str(e)}")


@router.get("/stats")
async def get_statistics():
    stats = vector_store.get_stats()
    doc_count = len(metadata_store.data)
    total_size = sum(meta.get('size', 0) for meta in metadata_store.data.values())
    return {
        "total_documents": doc_count,
        "total_chunks": stats.get('total_chunks', 0),
        "total_queries": config_store.get('total_queries'),
        "total_storage_size": total_size,
        "average_chunks_per_document": round(stats.get('total_chunks', 0) / max(1, doc_count), 2),
        "last_update": stats.get('last_update'),
    }


