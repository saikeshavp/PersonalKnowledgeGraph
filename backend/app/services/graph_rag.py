"""
GraphRAG Service
Implements hybrid retrieval combining vector similarity search and graph traversal.
This is the core GraphRAG engine for multi-hop reasoning and explainable answers.
Uses Google Gemini for LLM operations.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
import uuid

from app.database.neo4j_client import neo4j_client
from app.database.vector_store import vector_store


class GraphRAGEngine:
    """
    GraphRAG Engine - Hybrid retrieval system combining:
    1. Semantic similarity search (Vector Store)
    2. Knowledge graph traversal (Neo4j)
    3. Multi-hop reasoning
    4. Explainable answer generation
    
    """
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Groq client"""
        if self._client is None:
            # Robustly fetch key in case init happened before load_dotenv
            if not self.api_key:
                self.api_key = os.getenv("GROQ_API_KEY", "")
                
            try:
                from groq import Groq
                if self.api_key:
                    self._client = Groq(api_key=self.api_key)
                else:
                    print(f"Warning: GROQ_API_KEY not found. Ensure .env is loaded.")
            except ImportError:
                print("groq library not installed")
        return self._client
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using vector store's model"""
        try:
            return vector_store.generate_embeddings([text])[0]
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768
    
    # ============================================================
    # Core GraphRAG Query Methods
    # ============================================================
    
    def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Execute GraphRAG query.
        
        Args:
            question: Natural language question
            mode: Query mode - "vector", "graph", or "hybrid"
            top_k: Number of vector results
            max_hops: Maximum graph traversal depth
        
        Returns:
            Complete query response with answer, reasoning, and evidence
        """
        start_time = time.time()
        
        # Step 1: Vector similarity search
        vector_results = []
        if mode in ["vector", "hybrid"]:
            vector_results = self._vector_search(question, top_k)
        
        # Step 2: Extract entities from question
        question_entities = self._extract_question_entities(question)
        
        # Step 3: Graph traversal
        graph_context = {"entities": [], "relationships": []}
        if mode in ["graph", "hybrid"]:
            graph_context = self._graph_traversal(
                question_entities, 
                vector_results,
                max_hops
            )
        
        # Step 4: Combine contexts and generate answer
        answer, reasoning_path = self._generate_answer(
            question,
            vector_results,
            graph_context,
            mode
        )
        
        # Step 5: Prepare response
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "question": question,
            "answer": answer,
            "reasoning_path": reasoning_path,
            "evidence": self._format_evidence(vector_results),
            "graph_context": self._format_graph_context(graph_context),
            "mode_used": mode,
            "processing_time_ms": processing_time
        }
    
    def _vector_search(
        self, 
        question: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search"""
        try:
            # Use text search (ChromaDB handles embeddings internally)
            results = vector_store.search_by_text(question, top_k)
            return results
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def _extract_question_entities(
        self, 
        question: str
    ) -> List[str]:
        """Extract potential entities from the question using Gemini"""
        try:
            if not self.client:
                raise Exception("Groq client not initialized")

            prompt = f"""Extract key entities from this question. Return a JSON object with an 'entities' array of strings.
            
Question: {question}

Return ONLY valid JSON like: {{"entities": ["entity1", "entity2"]}}"""

            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(completion.choices[0].message.content)
            return result.get("entities", [])
            
        except Exception as e:
            # Fallback: extract capitalized words
            import re
            words = question.split()
            entities = []
            for word in words:
                clean = re.sub(r'[^\w]', '', word)
                if clean and (clean[0].isupper() or clean in ["who", "what", "where", "when", "how"]):
                    if len(clean) > 2:
                        entities.append(clean)
            return entities
    
    def _graph_traversal(
        self,
        question_entities: List[str],
        vector_results: List[Dict[str, Any]],
        max_hops: int
    ) -> Dict[str, Any]:
        """
        Perform multi-hop graph traversal from identified entities.
        """
        entities_to_explore = set()
        
        # Add entities from question
        for entity_name in question_entities:
            entities_to_explore.add(entity_name.lower())
        
        # Add entities mentioned in vector search results
        for result in vector_results:
            text = result.get("text", "").lower()
            for entity_name in question_entities:
                if entity_name.lower() in text:
                    entities_to_explore.add(entity_name.lower())
        
        # Find matching entities in the graph
        all_entities = neo4j_client.get_all_entities(limit=500)
        matching_entity_ids = []
        
        for entity in all_entities:
            entity_name = entity.get("name", "").lower()
            for search_term in entities_to_explore:
                if search_term in entity_name or entity_name in search_term:
                    matching_entity_ids.append(entity.get("id"))
                    break
        
        # Perform multi-hop traversal
        if matching_entity_ids:
            traversal_result = neo4j_client.multi_hop_traversal(
                matching_entity_ids[:10],  # Limit starting entities
                max_hops
            )
            return traversal_result
        
        return {"entities": [], "relationships": []}
    
    def _generate_answer(
        self,
        question: str,
        vector_results: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        mode: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate grounded answer using Gemini with context from both sources.
        """
        # Build context from vector results
        vector_context = ""
        for i, result in enumerate(vector_results[:5]):
            vector_context += f"\n[Source {i+1}]: {result.get('text', '')[:500]}"
        
        # Build context from graph
        graph_context_str = ""
        entities = graph_context.get("entities", [])
        for entity in entities[:20]:
            graph_context_str += f"\n- Entity: {entity.get('name', '')} (Type: {entity.get('type', '')})"
        
        # Create prompt for answer generation
        prompt = f"""You are a knowledge assistant using GraphRAG. Answer the question based on the provided context.

QUESTION: {question}

MODE: {mode}

DOCUMENT CONTEXT (from semantic search):
{vector_context if vector_context else "No relevant documents found."}

KNOWLEDGE GRAPH CONTEXT (entities and relationships):
{graph_context_str if graph_context_str else "No relevant graph entities found."}

Instructions:
1. Synthesize information from both contexts
2. Provide a clear, comprehensive answer
3. If information is insufficient, acknowledge it
4. Reference specific entities when possible

Provide your answer:"""
        
        try:
            if not self.client:
                raise Exception("Groq client not initialized")

            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful knowledge assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = completion.choices[0].message.content
            
            # Generate reasoning path
            reasoning_path = self._build_reasoning_path(
                question, vector_results, graph_context, answer
            )
            
            return answer, reasoning_path
            
        except Exception as e:
            return f"Unable to generate answer: {str(e)}", []
    
    def _build_reasoning_path(
        self,
        question: str,
        vector_results: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        answer: str
    ) -> List[Dict[str, Any]]:
        """Build step-by-step reasoning path for explainability"""
        reasoning = []
        step_num = 1
        
        # Step 1: Question analysis
        reasoning.append({
            "step_number": step_num,
            "description": f"Analyzed question to identify key concepts",
            "entities_involved": [],
            "relationships_used": [],
            "evidence": question
        })
        step_num += 1
        
        # Step 2: Vector search
        if vector_results:
            reasoning.append({
                "step_number": step_num,
                "description": f"Retrieved {len(vector_results)} relevant text chunks via semantic search",
                "entities_involved": [],
                "relationships_used": [],
                "evidence": f"Top result score: {vector_results[0].get('score', 0):.2f}"
            })
            step_num += 1
        
        # Step 3: Graph traversal
        entities = graph_context.get("entities", [])
        if entities:
            entity_names = [e.get("name", "") for e in entities[:5]]
            reasoning.append({
                "step_number": step_num,
                "description": f"Traversed knowledge graph, found {len(entities)} related entities",
                "entities_involved": entity_names,
                "relationships_used": [],
                "evidence": f"Key entities: {', '.join(entity_names)}"
            })
            step_num += 1
        
        # Step 4: Answer synthesis
        reasoning.append({
            "step_number": step_num,
            "description": "Synthesized answer from document and graph context using Gemini",
            "entities_involved": [],
            "relationships_used": [],
            "evidence": answer[:200] + "..." if len(answer) > 200 else answer
        })
        
        return reasoning
    
    def _format_evidence(
        self, 
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format vector search results as evidence"""
        evidence = []
        for result in vector_results:
            evidence.append({
                "source_doc": result.get("metadata", {}).get("source_doc", "Unknown"),
                "chunk_text": result.get("text", "")[:500],
                "relevance_score": result.get("score", 0),
                "entities": []  # Could be populated with entity mentions
            })
        return evidence
    
    def _format_graph_context(
        self, 
        graph_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format graph context for response"""
        nodes = []
        edges = []
        
        # Format entities as nodes
        for entity in graph_context.get("entities", []):
            nodes.append({
                "id": entity.get("id", ""),
                "name": entity.get("name", ""),
                "type": entity.get("type", ""),
                "source_doc": entity.get("source_doc", ""),
                "properties": {}
            })
        
        # Format relationships as edges
        for rel in graph_context.get("relationships", []):
            if isinstance(rel, dict):
                edges.append({
                    "id": rel.get("id", ""),
                    "source": rel.get("source_entity_id", ""),
                    "target": rel.get("target_entity_id", ""),
                    "relationship": rel.get("relationship_type", "RELATES_TO"),
                    "properties": {}
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }


# Singleton instance
graph_rag_engine = GraphRAGEngine()
