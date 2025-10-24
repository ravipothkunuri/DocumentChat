"""
API route handlers
"""
import json
import logging
import traceback
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import File, UploadFile, Request, HTTPException, APIRouter
from fastapi.responses import StreamingResponse

from backend.config import UPLOAD_DIR, VECTOR_DIR
from backend.models import (
    QueryRequest, DocumentUploadResponse, ModelConfig, DocumentInfo
)
from backend.utils import validate_file, get_available_models, check_ollama_health, clean_llm_response

logger = logging.getLogger(__name__)

router = APIRouter()


def init_managers(config_mgr, metadata_mgr, model_mgr, doc_processor, vec_store):
    """Initialize route dependencies"""
    global config_manager, metadata_manager, model_manager, document_processor, vector_store
    config_manager = config_mgr
    metadata_manager = metadata_mgr
    model_manager = model_mgr
    document_processor = doc_processor
    vector_store = vec_store


@router.get("/health", tags=["Health"])
async def health_check():
    """Check system health and configuration"""
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
                "model": config_manager.get('model'),
                "embedding_model": config_manager.get('embedding_model'),
                "chunk_size": config_manager.get('chunk_size'),
                "chunk_overlap": config_manager.get('chunk_overlap'),
                "temperature": config_manager.get('temperature')
            },
            "document_count": len(metadata_manager.metadata),
            "total_chunks": stats.get("total_chunks", 0),
            "total_queries": config_manager.get('total_queries')
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    logger.info(f"Upload request: {file.filename}")
    
    content = await file.read()
    file_size = len(content)
    
    validate_file(file.filename, file_size)
    
    if metadata_manager.exists(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Document '{file.filename}' already exists"
        )
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
        
        document_content = document_processor.load_document(file_path)
        documents = document_processor.split_text(document_content, file.filename)
        
        if not documents:
            raise ValueError("No content extracted from document")
        
        embeddings_model = model_manager.get_embeddings_model()
        texts = [doc["page_content"] for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        
        if not embeddings or len(embeddings) != len(documents):
            raise ValueError("Failed to generate embeddings")
        
        vector_store.add_documents(documents, embeddings)
        
        metadata_manager.add(file.filename, {
            "filename": file.filename,
            "size": file_size,
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
            file_size=file_size,
            message=f"Document processed successfully into {len(documents)} chunks"
        )
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        logger.debug(traceback.format_exc())
        
        if file_path.exists():
            file_path.unlink()
        metadata_manager.remove(file.filename)
        
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/query", tags=["Query"])
async def query_documents(request: Request, query: QueryRequest):
    """Query the document collection with streaming support"""
    logger.info(f"Query: '{query.question[:50]}...'")
    start_time = datetime.now()
    
    try:
        if vector_store.get_stats().get("total_chunks", 0) == 0:
            raise HTTPException(status_code=400, detail="No documents available. Upload documents first.")
        
        embeddings_model = model_manager.get_embeddings_model()
        query_embedding = embeddings_model.embed_query(query.question)
        similar_docs = vector_store.similarity_search(query_embedding, k=query.top_k)
        
        if not similar_docs:
            raise HTTPException(status_code=400, detail="No relevant documents found")
        
        context_parts = []
        sources = []
        similarity_scores = []
        
        for doc, score in similar_docs:
            context_parts.append(f"Source: {doc['metadata']['source']}\nContent: {doc['page_content']}")
            sources.append(doc['metadata']['source'])
            similarity_scores.append(float(score))
        
        context = "\n\n".join(context_parts)
        
        llm = model_manager.get_llm_model(query.model, query.temperature)
        
        unique_sources = list(set(sources))
        doc_identity = unique_sources[0] if len(unique_sources) == 1 else "your documents"
        prompt = f"""You are {doc_identity}, a helpful document assistant. Respond in first person.

Your content includes:
{context}

The user asks: {query.question}

Respond naturally as the document. Your response:"""
        
        if query.stream:
            async def generate():
                disconnected = False
                try:
                    metadata = {
                        "sources": sources,
                        "chunks_used": len(similar_docs),
                        "similarity_scores": similarity_scores,
                        "model_used": llm.model,
                        "endpoint_type": llm.endpoint_type,
                        "type": "metadata"
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"

                    for chunk in llm.stream(prompt):
                        if await request.is_disconnected():
                            disconnected = True
                            logger.info("Client disconnected during streaming")
                            break
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                    if not disconnected:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        completion = {"type": "done", "processing_time": processing_time}
                        yield f"data: {json.dumps(completion)}\n\n"
                        
                        config_manager.increment_queries()
                        logger.info(f"Query completed in {processing_time:.2f}s")
                    else:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.info(f"Query interrupted after {processing_time:.2f}s")
                except GeneratorExit:
                    logger.info("Generator closed by client disconnect")
                    raise
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    try:
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    except:
                        pass
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            answer = llm.invoke(prompt)
            answer = clean_llm_response(answer)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            config_manager.increment_queries()
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(similar_docs),
                "similarity_scores": similarity_scores,
                "processing_time": processing_time,
                "model_used": llm.model
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@router.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all uploaded documents"""
    return [DocumentInfo(**meta) for meta in metadata_manager.list_all()]


@router.get("/documents/{filename}/details", tags=["Documents"])
async def get_document_details(filename: str):
    """Get detailed information about a specific document"""
    if not metadata_manager.exists(filename):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    
    try:
        metadata = metadata_manager.get(filename)
        chunks_info = vector_store.get_chunks_by_source(filename)
        
        chunk_lengths = [chunk["metadata"].get("chunk_length", 0) for chunk in chunks_info]
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        
        return {
            "filename": metadata.get("filename"),
            "type": metadata.get("type"),
            "size": metadata.get("size"),
            "size_mb": round(metadata.get("size", 0) / (1024 * 1024), 3),
            "chunks": metadata.get("chunks"),
            "status": metadata.get("status"),
            "uploaded_at": metadata.get("uploaded_at"),
            "average_chunk_length": round(avg_chunk_length, 2),
            "total_chunk_length": sum(chunk_lengths),
            "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
            "chunk_preview": chunks_info[0]["page_content"][:200] + "..." if chunks_info else ""
        }
    except Exception as e:
        logger.error(f"Error getting details for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")


@router.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a specific document"""
    logger.info(f"Delete request: {filename}")
    
    if not metadata_manager.exists(filename):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    
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
        logger.error(f"Error deleting {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete("/clear", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents and embeddings"""
    try:
        vector_store.clear()
        metadata_manager.clear()
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        vector_store.save()
        
        logger.info("Cleared all documents")
        return {"message": "All documents cleared successfully", "cleared": True}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@router.post("/configure", tags=["Configuration"])
async def configure_system(config_update: ModelConfig):
    """Update system configuration"""
    try:
        changed_fields = config_manager.update(**config_update.dict(exclude_none=True))
        
        if "model" in changed_fields:
            model_manager.reset_llm_cache()
        
        if "embedding_model" in changed_fields:
            model_manager.reset_embeddings_model()
        
        logger.info(f"Configuration updated: {changed_fields}")
        
        response = {
            "message": "Configuration updated successfully",
            "changed_fields": changed_fields
        }
        
        if "embedding_model" in changed_fields:
            response["warning"] = "Embedding model changed. Consider rebuilding vectors."
        
        return response
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.get("/models", tags=["Configuration"])
async def get_models():
    """Get available models"""
    try:
        llm_models, embedding_models = get_available_models()
        
        return {
            "ollama": {
                "llm_models": llm_models,
                "embedding_models": embedding_models,
            },
            "current_config": {
                "model": config_manager.get('model'),
                "embedding_model": config_manager.get('embedding_model')
            }
        }
    except Exception as e:
        return {
            "ollama": {
                "llm_models": ["phi3", "llama3", "mistral", "deepseek-r1"],
                "embedding_models": ["nomic-embed-text"],
            },
            "current_config": {
                "model": config_manager.get('model'),
                "embedding_model": config_manager.get('embedding_model')
            }
        }


@router.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get system statistics"""
    try:
        stats = vector_store.get_stats()
        doc_count = len(metadata_manager.metadata)
        total_size = sum(meta.get('size', 0) for meta in metadata_manager.metadata.values())
        avg_chunks = stats.get('total_chunks', 0) / max(1, doc_count)
        
        return {
            "total_documents": doc_count,
            "total_chunks": stats.get('total_chunks', 0),
            "total_queries": config_manager.get('total_queries'),
            "total_storage_size": total_size,
            "average_chunks_per_document": round(avg_chunks, 2),
            "last_update": stats.get('last_update'),
            "vector_store_size_mb": round(stats.get('vector_store_size', 0) / (1024 * 1024), 2)
        }
    except Exception:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "total_storage_size": 0,
            "average_chunks_per_document": 0.0
        }


@router.post("/rebuild-vectors", tags=["Maintenance"])
async def rebuild_vectors():
    """Rebuild all vectors with current embedding model"""
    logger.info("Rebuild vectors request")
    
    try:
        vector_store.clear()
        results = {}
        embeddings_model = model_manager.get_embeddings_model()
        
        for filename in list(metadata_manager.metadata.keys()):
            try:
                file_path = UPLOAD_DIR / filename
                if not file_path.exists():
                    results[filename] = {"success": False, "error": "File not found"}
                    continue
                
                document_content = document_processor.load_document(file_path)
                documents = document_processor.split_text(document_content, filename)
                
                texts = [doc["page_content"] for doc in documents]
                embeddings = embeddings_model.embed_documents(texts)
                
                vector_store.add_documents(documents, embeddings)
                
                metadata = metadata_manager.get(filename) or {}
                metadata["chunks"] = len(documents)
                metadata_manager.add(filename, metadata)
                
                results[filename] = {"success": True, "chunks": len(documents)}
            except Exception as e:
                logger.error(f"Error rebuilding {filename}: {e}")
                results[filename] = {"success": False, "error": str(e)}
        
        vector_store.save()
        
        success_count = sum(1 for result in results.values() if result["success"])
        
        logger.info(f"Rebuild completed: {success_count}/{len(results)} successful")
        
        return {
            "message": f"Rebuild completed: {success_count}/{len(results)} documents processed",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error rebuilding vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild vectors: {str(e)}")
