"""
Neo4j Database Client
Handles all Neo4j graph database operations for the knowledge graph.
"""

import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from contextlib import contextmanager


class Neo4jClient:
    """Neo4j database client for knowledge graph operations"""
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self._driver = None
    
    @property
    def driver(self):
        """Lazy initialization of Neo4j driver"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
        return self._driver
    
    def _sanitize_value(self, value):
        """Convert Neo4j types to standard Python types"""
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        if isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        return value

    def _sanitize_record(self, record):
        """Sanitize a Neo4j record or node dictionary"""
        if not record:
            return None
        if hasattr(record, 'items'):
             return {k: self._sanitize_value(v) for k, v in record.items()}
        return record
        
    def close(self):
        """Close the driver connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    @contextmanager
    def session(self):
        """Context manager for Neo4j session"""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check if Neo4j is connected"""
        try:
            with self.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False
    
    # ============================================================
    # Entity Operations
    # ============================================================
    
    def create_entity(
        self, 
        entity_id: str,
        name: str, 
        entity_type: str, 
        source_doc: str, 
        chunk_id: str,
        properties: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create an entity node in the graph"""
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.source_doc = $source_doc,
            e.chunk_id = $chunk_id,
            e.created_at = datetime()
        """
        
        if properties:
            for key, value in properties.items():
                query += f", e.{key} = ${key}"
        
        query += " RETURN e"
        
        params = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "source_doc": source_doc,
            "chunk_id": chunk_id,
            **(properties or {})
        }
        
        with self.session() as session:
            result = session.run(query, params)
            record = result.single()
            return self._sanitize_record(dict(record["e"])) if record else None
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID"""
        query = "MATCH (e:Entity {id: $id}) RETURN e"
        
        with self.session() as session:
            result = session.run(query, {"id": entity_id})
            record = result.single()
            return self._sanitize_record(dict(record["e"])) if record else None
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type"""
        query = "MATCH (e:Entity {type: $type}) RETURN e"
        
        with self.session() as session:
            result = session.run(query, {"type": entity_type})
            return [self._sanitize_record(dict(record["e"])) for record in result]
    
    def get_all_entities(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all entities with optional limit"""
        query = "MATCH (e:Entity) RETURN e LIMIT $limit"
        
        with self.session() as session:
            result = session.run(query, {"limit": limit})
            return [self._sanitize_record(dict(record["e"])) for record in result]
    
    # ============================================================
    # Relationship Operations
    # ============================================================
    
    def create_relationship(
        self,
        rel_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        source_doc: str,
        chunk_id: str,
        properties: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a relationship between two entities"""
        # Sanitize relationship type for Cypher (remove spaces, use uppercase)
        safe_rel_type = relationship_type.upper().replace(" ", "_").replace("-", "_")
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{safe_rel_type} {{id: $rel_id}}]->(target)
        SET r.relationship = $relationship_type,
            r.source_doc = $source_doc,
            r.chunk_id = $chunk_id,
            r.created_at = datetime()
        RETURN source, r, target
        """
        
        params = {
            "rel_id": rel_id,
            "source_id": source_entity_id,
            "target_id": target_entity_id,
            "relationship_type": relationship_type,
            "source_doc": source_doc,
            "chunk_id": chunk_id
        }
        
        with self.session() as session:
            result = session.run(query, params)
            record = result.single()
            if record:
                return {
                    "source": self._sanitize_record(dict(record["source"])),
                    "relationship": self._sanitize_record(dict(record["r"])),
                    "target": self._sanitize_record(dict(record["target"]))
                }
            return None
    
    def get_all_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all relationships"""
        query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        RETURN source, r, target
        LIMIT $limit
        """
        
        with self.session() as session:
            result = session.run(query, {"limit": limit})
            relationships = []
            for record in result:
                relationships.append({
                    "source": self._sanitize_record(dict(record["source"])),
                    "relationship": self._sanitize_record(dict(record["r"])),
                    "target": self._sanitize_record(dict(record["target"]))
                })
            return relationships
    
    # ============================================================
    # Graph Traversal Operations
    # ============================================================
    
    def get_entity_neighbors(
        self, 
        entity_id: str, 
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get all entities within N hops of the given entity"""
        query = f"""
        MATCH path = (e:Entity {{id: $id}})-[*1..{max_depth}]-(neighbor:Entity)
        RETURN DISTINCT neighbor, length(path) as distance
        ORDER BY distance
        """
        
        with self.session() as session:
            result = session.run(query, {"id": entity_id})
            neighbors = []
            for record in result:
                neighbors.append({
                    "entity": self._sanitize_record(dict(record["neighbor"])),
                    "distance": record["distance"]
                })
            return neighbors
    
    def find_path_between_entities(
        self, 
        source_id: str, 
        target_id: str, 
        max_depth: int = 4
    ) -> List[Dict[str, Any]]:
        """Find shortest path between two entities"""
        query = f"""
        MATCH path = shortestPath(
            (source:Entity {{id: $source_id}})-[*..{max_depth}]-(target:Entity {{id: $target_id}})
        )
        RETURN path
        """
        
        with self.session() as session:
            result = session.run(query, {
                "source_id": source_id,
                "target_id": target_id
            })
            record = result.single()
            if record:
                path = record["path"]
                return self._parse_path(path)
            return []
    
    def multi_hop_traversal(
        self,
        start_entities: List[str],
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Perform multi-hop traversal from multiple starting entities.
        Returns all reachable nodes and edges within max_hops.
        """
        query = f"""
        UNWIND $entity_ids as entity_id
        MATCH (start:Entity {{id: entity_id}})
        OPTIONAL MATCH path = (start)-[*1..{max_hops}]-(connected:Entity)
        WITH start, connected, relationships(path) as rels
        RETURN DISTINCT 
            start,
            collect(DISTINCT connected) as connected_entities,
            collect(DISTINCT rels) as all_relationships
        """
        
        with self.session() as session:
            result = session.run(query, {"entity_ids": start_entities})
            
            all_entities = set()
            all_rels = []
            
            for record in result:
                if record["start"]:
                    all_entities.add(frozenset(self._sanitize_record(dict(record["start"])).items()))
                for entity in record["connected_entities"]:
                    if entity:
                        all_entities.add(frozenset(self._sanitize_record(dict(entity)).items()))
            
            return {
                "entities": [dict(e) for e in all_entities],
                "relationships": [self._sanitize_record(r) for r in all_rels]
            }
    
    def _parse_path(self, path) -> List[Dict[str, Any]]:
        """Parse a Neo4j path into a list of nodes and relationships"""
        result = []
        nodes = list(path.nodes)
        rels = list(path.relationships)
        
        for i, node in enumerate(nodes):
            result.append({
                "type": "node",
                "data": self._sanitize_record(dict(node))
            })
            if i < len(rels):
                result.append({
                    "type": "relationship",
                    "data": self._sanitize_record(dict(rels[i]))
                })
        
        return result
    
    # ============================================================
    # Document Tracking
    # ============================================================
    
    def create_document(
        self, 
        doc_id: str, 
        filename: str,
        file_type: str = None
    ) -> Dict[str, Any]:
        """Create a document node for provenance tracking"""
        if file_type is None:
            file_type = filename.split(".")[-1].lower() if "." in filename else "unknown"
        
        query = """
        MERGE (d:Document {id: $id})
        SET d.filename = $filename,
            d.file_type = $file_type,
            d.upload_date = datetime()
        RETURN d
        """
        
        with self.session() as session:
            result = session.run(query, {"id": doc_id, "filename": filename, "file_type": file_type})
            record = result.single()
            return self._sanitize_record(dict(record["d"])) if record else None
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the graph"""
        query = """
        MATCH (e:Entity)
        WITH count(e) as entity_count
        MATCH ()-[r]->()
        WITH entity_count, count(r) as relationship_count
        MATCH (d:Document)
        RETURN entity_count, relationship_count, count(d) as document_count
        """
        
        with self.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                return {
                    "entities": record["entity_count"],
                    "relationships": record["relationship_count"],
                    "documents": record["document_count"]
                }
            return {"entities": 0, "relationships": 0, "documents": 0}
    
    def clear_graph(self):
        """Clear all data from the graph (use with caution!)"""
        query = "MATCH (n) DETACH DELETE n"
        
        with self.session() as session:
            session.run(query)
    def clear_database(self):
        """Delete all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    # ============================================================
    # File Explorer Operations
    # ============================================================
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all document nodes"""
        query = """
        MATCH (d:Document)
        RETURN d
        ORDER BY d.upload_date DESC
        """
        
        with self.session() as session:
            result = session.run(query)
            return [self._sanitize_record(dict(record["d"])) for record in result]
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        query = "MATCH (d:Document {id: $id}) RETURN d"
        
        with self.session() as session:
            result = session.run(query, {"id": doc_id})
            record = result.single()
            return self._sanitize_record(dict(record["d"])) if record else None
    
    def get_doc_id_by_filename(self, filename: str) -> Optional[str]:
        """Get document ID by filename"""
        query = "MATCH (d:Document {filename: $filename}) RETURN d.id as id"
        
        with self.session() as session:
            result = session.run(query, {"filename": filename})
            record = result.single()
            return record["id"] if record else None
    
    def get_entity_count_by_doc(self, doc_id: str) -> int:
        """Get count of entities for a specific document"""
        # Get the filename first
        doc = self.get_document(doc_id)
        if not doc:
            return 0
        
        filename = doc.get("filename", "")
        query = """
        MATCH (e:Entity {source_doc: $filename})
        RETURN count(e) as count
        """
        
        with self.session() as session:
            result = session.run(query, {"filename": filename})
            record = result.single()
            return record["count"] if record else 0
    
    def get_entities_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific document"""
        doc = self.get_document(doc_id)
        if not doc:
            return []
        
        filename = doc.get("filename", "")
        query = """
        MATCH (e:Entity {source_doc: $filename})
        RETURN e
        """
        
        with self.session() as session:
            result = session.run(query, {"filename": filename})
            return [self._sanitize_record(dict(record["e"])) for record in result]
    
    def get_relationships_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a specific document"""
        doc = self.get_document(doc_id)
        if not doc:
            return []
        
        filename = doc.get("filename", "")
        query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        WHERE r.source_doc = $filename
        RETURN r.id as id, r.relationship as relationship_type, 
               source.name as source_name, target.name as target_name
        """
        
        with self.session() as session:
            result = session.run(query, {"filename": filename})
            relationships = []
            for record in result:
                relationships.append({
                    "id": record["id"],
                    "relationship_type": record["relationship_type"],
                    "source_name": record["source_name"],
                    "target_name": record["target_name"]
                })
            return relationships
    
    # Common English stop words to filter out from entity searches
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
        "are", "been", "has", "have", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall", "not",
        "no", "so", "if", "then", "than", "that", "this", "these", "those",
        "my", "your", "his", "her", "its", "our", "their", "me", "him",
        "us", "them", "who", "whom", "which", "what", "where", "when",
        "how", "why", "all", "each", "every", "both", "few", "more",
        "some", "any", "most", "other", "into", "over", "after", "before",
        "between", "under", "about", "up", "out", "off", "down", "just",
        "now", "here", "there", "also", "very", "only", "even", "still",
        "find", "show", "get", "give", "tell", "notes", "note", "file",
        "files", "document", "documents", "search", "look", "need",
    }

    def search_entities_by_name(self, query: str) -> List[Dict[str, Any]]:
        """Search entities by name (case-insensitive partial match)"""
        # Split query into words, filter out stop words and short words
        words = [
            w for w in query.lower().split()
            if w not in self.STOP_WORDS and len(w) >= 2
        ]
        
        if not words:
            return []
        
        cypher_query = """
        MATCH (e:Entity)
        WHERE any(word IN $words WHERE toLower(e.name) CONTAINS word)
        RETURN e
        LIMIT 50
        """
        
        with self.session() as session:
            result = session.run(cypher_query, {"words": words})
            return [self._sanitize_record(dict(record["e"])) for record in result]


# Singleton instance
neo4j_client = Neo4jClient()
