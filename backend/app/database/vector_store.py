"""
Vector Store Client
Handles vector embeddings storage and similarity search using ChromaDB.
"""

import os
import re
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStoreClient:
    """ChromaDB client for vector embeddings and similarity search"""
    
    # Use a high-quality embedding model (768-dim, best general-purpose model)
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    EMBEDDING_DIM = 768
    
    def __init__(self):
        self.host = os.getenv("CHROMA_HOST", "localhost")
        self.port = int(os.getenv("CHROMA_PORT", "8001"))
        self._client = None
        self._collection = None
        self._embedding_function = None
    
    @property
    def client(self):
        """Lazy initialization of ChromaDB client"""
        if self._client is None:
            try:
                # Try to connect to ChromaDB server with longer timeout
                import httpx
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(
                        anonymized_telemetry=False,
                        chroma_client_auth_provider=None
                    )
                )
                # Test connection
                self._client.heartbeat()
            except Exception as e:
                print(f"ChromaDB HTTP connection failed: {e}")
                # Fall back to persistent local client
                self._client = chromadb.PersistentClient(
                    path="./chroma_data",
                    settings=Settings(anonymized_telemetry=False)
                )
        return self._client
    
    @property
    def collection(self):
        """Get or create the main collection with cosine distance metric"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="knowledge_chunks",
                metadata={
                    "description": "Document chunks for knowledge graph",
                    "hnsw:space": "cosine"  # Use cosine similarity instead of L2
                }
            )
        return self._collection
    
    def health_check(self) -> bool:
        """Check if ChromaDB is connected"""
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False
    
    # ============================================================
    # Text Preprocessing
    # ============================================================
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Clean and normalize text before embedding for better similarity results.
        """
        # Remove null bytes and control characters (except newlines/tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse multiple whitespace/newlines into single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    # ============================================================
    # Embedding Operations
    # ============================================================
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """
        Add document chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'text', 'doc_id', 'chunk_index'
            embeddings: List of embedding vectors corresponding to chunks
        """
        try:
            ids = [chunk["id"] for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "source_doc": chunk.get("source_doc", ""),
                    "char_count": len(chunk["text"])
                }
                for chunk in chunks
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Error adding chunks: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            List of matching chunks with scores
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            chunks = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    # With cosine distance: score = 1 - distance (gives 0-1 similarity)
                    score = max(0.0, 1.0 - distance)
                    chunks.append({
                        "id": chunk_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": distance,
                        "score": score
                    })
            
            return chunks
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using text query.
        Preprocesses the query and generates embeddings with the same model used during indexing.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
        """
        try:
            # Preprocess query text the same way we preprocess chunks
            clean_query = self.preprocess_text(query_text)
            query_embedding = self.generate_embeddings([clean_query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            chunks = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    score = max(0.0, 1.0 - distance)
                    chunks.append({
                        "id": chunk_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": distance,
                        "score": score
                    })
            
            return chunks
        except Exception as e:
            print(f"Error searching by text: {e}")
            return []
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID"""
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "text": results["documents"][0] if results["documents"] else "",
                    "metadata": results["metadatas"][0] if results["metadatas"] else {}
                }
            return None
        except Exception as e:
            print(f"Error getting chunk: {e}")
            return None
    
    def get_chunks_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    chunks.append({
                        "id": chunk_id,
                        "text": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    })
            
            return sorted(chunks, key=lambda x: x["metadata"].get("chunk_index", 0))
        except Exception as e:
            print(f"Error getting chunks by doc: {e}")
            return []
    
    def delete_chunks_by_doc(self, doc_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            self.collection.delete(where={"doc_id": doc_id})
            return True
        except Exception as e:
            print(f"Error deleting chunks: {e}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {"total_chunks": count}
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chunks": 0}
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using local SentenceTransformer.
        Uses all-mpnet-base-v2 (768-dim) for high-quality semantic embeddings.
        Preprocesses texts before embedding for consistency.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            if self._embedding_function is None:
                print(f"Loading local embedding model ({self.EMBEDDING_MODEL})...")
                self._embedding_function = SentenceTransformer(self.EMBEDDING_MODEL)
            
            # Preprocess all texts before embedding
            clean_texts = [self.preprocess_text(t) for t in texts]
            
            # Generate embeddings
            embeddings = self._embedding_function.encode(
                clean_texts,
                normalize_embeddings=True  # L2-normalize for cosine similarity
            ).tolist()
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise  # Don't fall back to dummy embeddings — let the caller handle it
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection("knowledge_chunks")
            self._collection = None
        except Exception as e:
            print(f"Error clearing collection: {e}")


# Singleton instance
vector_store = VectorStoreClient()
