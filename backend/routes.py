"""API Routes - 240 lines"""
import json
import logging
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse

from backend.models import QueryRequest, DocumentUploadResponse, DocumentInfo
from backend.config import UPLOAD_DIR, FIXED_MODEL
from backend.utils import check_ollama_health, validate_file, clean_llm_response
from backend.config_manager import ConfigManager
from backend.metadata_manager import MetadataManager
from backend.model_manager import ModelManager
from backend.document_processor import DocumentProcessor
from vector_store import VectorStore

logger = logging.getLogger(__name__)

# Initialize managers
config_manager = ConfigManager()
metadata_manager = MetadataManager()
vector_store = VectorStore()
model_manager = ModelManager(config_manager)
document_processor = DocumentProcessor(config_manager)

# Load vector store
try:
    vector_store.load()
    logger.info("Vector store loaded")
except Exception as e:
    logger.warning(f"Could not load vector store: {e}")

# Create router
router = APIRouter()

@router.get("/health")
async def health_check():
    """System health check"""
    try:
        ollama_available, ollama_message = check_ollama_health()
        stats = vector_store.get_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ollama_status": {
                "available": ollama_available,
                "message": ollama_message
            },
            "configuration": {
                "model": FIXED_MODEL,
                "embedding_model": config_manager.get('embedding_model'),
                "chunk_size": config_manager.get('chunk_size')
            },
            "document_count": len(metadata_manager.metadata),
            "total_chunks": stats.get("total_chunks", 0),
            "total_queries": config_manager.get('total_queries')
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    logger.info(f"Upload request: {file.filename}")
    
    content = await file.read()
    validate_file(file.filename, len(content))
    
    if metadata_manager.exists(file.filename):
        raise HTTPException(400, f"Document '{file.filename}' already exists")
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Process document
        doc_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(doc_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        embeddings_model = model_manager.get_embeddings_model()
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        # Store in vector store
        vector_store.add_documents(documents, embeddings)
        
        # Save metadata
        metadata_manager.add(file.filename, {
            "filename": file.filename,
            "size": len(content),
            "chunks": len(documents),
            "status": "processed",
            "uploaded_at": datetime.now().isoformat(),
            "type": file_path.suffix[1:].lower()
        })
        
        vector_store.save()
        logger.info(f"Processed {file.filename}: {len(documents)} chunks")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            file_size=len(content),
            message=f"Document processed into {len(documents)} chunks"
        )
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        metadata_manager.remove(file.filename)
        raise HTTPException(500, f"Failed to process document: {str(e)}")

@router.post("/query")
async def query_documents(request: Request, query: QueryRequest):
    """Query documents with async streaming"""
    logger.info(f"Query: '{query.question[:50]}...'")
    start_time = datetime.now()
    
    try:
        if vector_store.get_stats().get("total_chunks", 0) == 0:
            raise HTTPException(400, "No documents available. Upload documents first.")
        
        # Get embeddings and search
        embeddings_model = model_manager.get_embeddings_model()
        query_embedding = embeddings_model.embed_query(query.question)
        similar_docs = vector_store.similarity_search(query_embedding, k=query.top_k)
        
        if not similar_docs:
            raise HTTPException(400, "No relevant documents found")
        
        # Build context
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
        
        # Build prompt
        prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person.

Your content:
{context}

User asks: {query.question}

Your response:"""
        
        llm = model_manager.get_llm()
        
        if query.stream:
            async def generate():
                try:
                    # Send metadata
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": scores,
                        "model_used": FIXED_MODEL,
                        "type": "metadata"
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"
                    
                    # Stream content
                    async for chunk in llm.astream(prompt):
                        if await request.is_disconnected():
                            logger.info("Client disconnected")
                            break
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                    
                    # Send completion
                    processing_time = (datetime.now() - start_time).total_seconds()
                    yield f"data: {json.dumps({'type': 'done', 'processing_time': processing_time})}\n\n"
                    
                    config_manager.increment_queries()
                    logger.info(f"Query completed in {processing_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Non-streaming response
            answer = await llm.ainvoke(prompt)
            answer = clean_llm_response(answer)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            config_manager.increment_queries()
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(similar_docs),
                "similarity_scores": scores,
                "processing_time": processing_time,
                "model_used": FIXED_MODEL
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, f"Failed to process query: {str(e)}")

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    return [DocumentInfo(**meta) for meta in metadata_manager.list_all()]

@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a specific document"""
    logger.info(f"Delete request: {filename}")
    
    if not metadata_manager.exists(filename):
        raise HTTPException(404, f"Document '{filename}' not found")
    
    try:
        vector_store.remove_documents_by_source(filename)
        metadata_manager.remove(filename)
        
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        vector_store.save()
        logger.info(f"Deleted: {filename}")
        return {"message": f"Document '{filename}' deleted successfully"}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(500, f"Failed to delete document: {str(e)}")

@router.delete("/clear")
async def clear_all_documents():
    """Clear all documents and embeddings"""
    try:
        vector_store.clear()
        metadata_manager.metadata = {}
        metadata_manager.save()
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        vector_store.save()
        logger.info("Cleared all documents")
        return {"message": "All documents cleared successfully", "cleared": True}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(500, f"Failed to clear documents: {str(e)}")

@router.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = vector_store.get_stats()
        doc_count = len(metadata_manager.metadata)
        total_size = sum(meta.get('size', 0) for meta in metadata_manager.metadata.values())
        
        return {
            "total_documents": doc_count,
            "total_chunks": stats.get('total_chunks', 0),
            "total_queries": config_manager.get('total_queries'),
            "total_storage_size": total_size,
            "average_chunks_per_document": round(stats.get('total_chunks', 0) / max(1, doc_count), 2),
            "last_update": stats.get('last_update')
        }
    except Exception:
        return {"total_documents": 0, "total_chunks": 0, "total_queries": 0}

# Export model manager for cleanup
def get_model_manager():
    return model_manager
