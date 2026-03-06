"""
AI File Explorer API Endpoints
Search and retrieve uploaded files using GraphRAG + semantic search.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Optional
from pathlib import Path
import os

from app.models.schemas import (
    FileSearchRequest, FileSearchResponse, FileInfo, FileMatch
)
from app.database.neo4j_client import neo4j_client
from app.database.vector_store import vector_store


router = APIRouter()


@router.get("/files", response_model=List[FileInfo])
async def list_files(
    file_type: Optional[str] = Query(None, description="Filter by file type (pdf, docx, txt)")
):
    """
    List all uploaded files with metadata.
    """
    try:
        documents = neo4j_client.get_all_documents()
        
        # Batch-fetch all entity counts in one pass to avoid N+1 queries
        entity_counts = {}
        for doc in documents:
            entity_counts[doc["id"]] = 0
        
        # Single query to get all entity counts grouped by source_doc
        all_entities = neo4j_client.get_all_entities(limit=5000)
        filename_to_doc_id = {doc.get("filename", ""): doc["id"] for doc in documents}
        for entity in all_entities:
            source_doc = entity.get("source_doc", "")
            doc_id = filename_to_doc_id.get(source_doc)
            if doc_id:
                entity_counts[doc_id] = entity_counts.get(doc_id, 0) + 1
        
        files = []
        uploads_dir = Path("./uploads")
        
        for doc in documents:
            doc_id = doc["id"]
            filename = doc.get("filename", "Unknown")
            doc_file_type = doc.get("file_type", filename.split(".")[-1].lower() if "." in filename else "unknown")
            
            # Apply file type filter early
            if file_type and doc_file_type != file_type.lower():
                continue
            
            # Get file size from disk
            file_size = None
            try:
                matching = list(uploads_dir.glob(f"{doc_id}_*"))
                if matching:
                    file_size = matching[0].stat().st_size
            except Exception:
                pass
            
            file_info = FileInfo(
                file_id=doc_id,
                filename=filename,
                file_type=doc_file_type,
                upload_date=doc.get("upload_date", ""),
                entity_count=entity_counts.get(doc_id, 0),
                status="complete"
            )
            
            files.append(file_info)
        
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/search", response_model=FileSearchResponse)
async def search_files(request: FileSearchRequest):
    """
    AI-powered file search using GraphRAG.
    
    Combines semantic search over document chunks with knowledge graph
    traversal to find relevant files based on natural language queries.
    """
    try:
        query = request.query
        top_k = request.top_k
        file_type_filter = request.file_type
        
        # Step 1: Semantic search over uploaded chunks
        semantic_results = vector_store.search_by_text(query, top_k=top_k * 3)
        
        # Step 2: Get entities matching the query terms
        matching_entities = neo4j_client.search_entities_by_name(query)
        
        # Step 3: Aggregate scores per file_id
        file_scores = {}
        file_matched_chunks = {}
        file_matched_entities = {}
        file_chunk_counts = {}  # Track chunk count for averaging
        
        # Build a local cache for filename -> doc_id lookups to avoid repeated DB calls
        _filename_to_doc_id_cache = {}

        def _cached_get_doc_id(filename: str) -> Optional[str]:
            if filename not in _filename_to_doc_id_cache:
                _filename_to_doc_id_cache[filename] = neo4j_client.get_doc_id_by_filename(filename)
            return _filename_to_doc_id_cache[filename]

        def _ensure_file_entry(doc_id: str):
            """Initialize tracking dicts for a new doc_id."""
            if doc_id not in file_scores:
                file_scores[doc_id] = {
                    "semantic_score": 0.0,
                    "entity_score": 0.0,
                    "graph_score": 0.0
                }
                file_matched_chunks[doc_id] = []
                file_matched_entities[doc_id] = set()
                file_chunk_counts[doc_id] = 0
        
        # Process semantic search results
        for result in semantic_results:
            doc_id = result["metadata"].get("doc_id", "")
            if not doc_id:
                continue
            
            _ensure_file_entry(doc_id)
            
            # Add semantic similarity score (accumulate for averaging)
            score = result.get("score", 0)
            file_scores[doc_id]["semantic_score"] += score
            file_chunk_counts[doc_id] += 1
            
            # Store matched chunk text (limit to first 200 chars)
            chunk_preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            file_matched_chunks[doc_id].append({
                "text": chunk_preview,
                "score": score
            })
        
        # Average the semantic scores instead of summing
        for doc_id in file_scores:
            count = file_chunk_counts.get(doc_id, 1)
            if count > 0:
                file_scores[doc_id]["semantic_score"] /= count
        
        # Process matching entities and their related documents
        for entity in matching_entities:
            source_doc = entity.get("source_doc", "")
            doc_id = _cached_get_doc_id(source_doc)
            
            if not doc_id:
                continue
            
            _ensure_file_entry(doc_id)
            file_scores[doc_id]["entity_score"] += 1.0
            file_matched_entities[doc_id].add(entity["name"])
            
            # Step 4: Graph traversal - find related entities
            neighbors = neo4j_client.get_entity_neighbors(entity["id"], max_depth=2)
            for neighbor in neighbors:
                neighbor_entity = neighbor.get("entity", {})
                neighbor_doc = neighbor_entity.get("source_doc", "")
                neighbor_doc_id = _cached_get_doc_id(neighbor_doc)
                
                if not neighbor_doc_id:
                    continue
                
                _ensure_file_entry(neighbor_doc_id)
                
                # Graph relationship score (weighted by distance)
                distance = neighbor.get("distance", 1)
                file_scores[neighbor_doc_id]["graph_score"] += 1.0 / distance
                file_matched_entities[neighbor_doc_id].add(neighbor_entity.get("name", ""))
        
        # Step 5: Calculate final scores and rank files
        ranked_files = []
        max_score = 0.0
        file_raw_scores = {}
        
        # Normalize entity_score to 0-1 range
        max_entity_score = max((s["entity_score"] for s in file_scores.values()), default=1.0) or 1.0
        max_graph_score = max((s["graph_score"] for s in file_scores.values()), default=1.0) or 1.0
        
        for doc_id, scores in file_scores.items():
            # Weighted combination of normalized scores
            final_score = (
                scores["semantic_score"] * 0.5 +                          # Already 0-1 from cosine
                (scores["entity_score"] / max_entity_score) * 0.3 +       # Normalized entity score
                (scores["graph_score"] / max_graph_score) * 0.2           # Normalized graph score
            )
            file_raw_scores[doc_id] = final_score
            if final_score > max_score:
                max_score = final_score
        
        for doc_id, scores in file_scores.items():
            # Normalize score to 0-1 range
            raw_score = file_raw_scores[doc_id]
            normalized_score = raw_score / max_score if max_score > 0 else 0.0
            
            # Get document info
            doc_info = neo4j_client.get_document(doc_id)
            if not doc_info:
                continue
            
            filename = doc_info.get("filename", "Unknown")
            file_type = filename.split(".")[-1].lower() if "." in filename else "unknown"
            
            # Apply file type filter
            if file_type_filter and file_type != file_type_filter.lower():
                continue
            
            # Get relationships for this file
            relationships = neo4j_client.get_relationships_by_doc(doc_id)
            relationship_types = list(set([r.get("relationship_type", "") for r in relationships]))[:5]
            
            # Only include files with meaningful relevance
            if normalized_score < 0.2:
                continue
            
            ranked_files.append(FileMatch(
                file_id=doc_id,
                filename=filename,
                file_type=file_type,
                upload_date=doc_info.get("upload_date", ""),
                relevance_score=round(normalized_score, 4),
                matched_entities=list(file_matched_entities.get(doc_id, set()))[:10],
                matched_relationships=relationship_types,
                matched_chunks=file_matched_chunks.get(doc_id, [])[:3],
                explanation=_generate_match_explanation(
                    scores, 
                    list(file_matched_entities.get(doc_id, set())),
                    relationship_types
                )
            ))
        
        # Sort by relevance score
        ranked_files.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit to top_k results
        ranked_files = ranked_files[:top_k]
        
        return FileSearchResponse(
            query=query,
            total_matches=len(ranked_files),
            files=ranked_files
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """
    Get detailed information about a specific file.
    """
    try:
        doc = neo4j_client.get_document(file_id)
        if not doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get all entities for this file
        entities = neo4j_client.get_entities_by_doc(file_id)
        
        # Get all relationships for this file
        relationships = neo4j_client.get_relationships_by_doc(file_id)
        
        # Get chunks
        chunks = vector_store.get_chunks_by_doc(file_id)
        
        # Get file size from disk
        file_size = None
        uploads_dir = Path("./uploads")
        try:
            matching = list(uploads_dir.glob(f"{file_id}_*"))
            if matching:
                file_size = matching[0].stat().st_size
        except Exception:
            pass
        
        return {
            "file_id": file_id,
            "filename": doc.get("filename", "Unknown"),
            "file_type": doc.get("filename", "").split(".")[-1].lower(),
            "upload_date": doc.get("upload_date", ""),
            "file_size": file_size,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "chunk_count": len(chunks),
            "entities": entities[:50],  # Limit for response size
            "relationships": relationships[:50]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}/download")
async def download_file(file_id: str):
    """
    Download the original uploaded file.
    """
    try:
        doc = neo4j_client.get_document(file_id)
        if not doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        filename = doc.get("filename", "")
        
        # Find the file in uploads directory
        uploads_dir = Path("./uploads")
        file_pattern = f"{file_id}_*"
        
        matching_files = list(uploads_dir.glob(file_pattern))
        
        if not matching_files:
            # Try exact filename match
            matching_files = list(uploads_dir.glob(f"*{filename}"))
        
        if not matching_files:
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        file_path = matching_files[0]
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_match_explanation(
    scores: dict,
    entities: list,
    relationships: list
) -> str:
    """Generate a human-readable explanation for why a file matched."""
    parts = []
    
    if scores["semantic_score"] > 0:
        parts.append(f"Content similarity: {scores['semantic_score']:.2f}")
    
    if entities:
        entity_str = ", ".join(entities[:3])
        if len(entities) > 3:
            entity_str += f" (+{len(entities) - 3} more)"
        parts.append(f"Matched entities: {entity_str}")
    
    if relationships:
        rel_str = ", ".join(relationships[:3])
        parts.append(f"Related through: {rel_str}")
    
    if scores["graph_score"] > 0:
        parts.append(f"Graph connections found")
    
    return " | ".join(parts) if parts else "Matched based on content"
