"""
Face clustering service for the facial recognition system.

This module provides face clustering functionality using vector search
to group similar faces into unique persons.

NOTE: This is System 2's clustering service - KEPT UNCHANGED for proven reliability.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from bson import ObjectId

from src.config import get_settings


class FaceClusteringService:
    """
    Service for clustering face embeddings into unique persons.
    
    Handles vector search, similarity analysis, and face grouping operations.
    """
    
    def __init__(self, db_service):
        """
        Initialize the face clustering service.
        
        Args:
            db_service: Database service instance
        """
        self.db_service = db_service
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Clustering configuration
        self.similarity_threshold = self.settings.SIMILARITY_THRESHOLD
        self.vector_index_name = self.settings.VECTOR_INDEX_NAME
        
        self.logger.info(f"🧩 Face clustering initialized with threshold: {self.similarity_threshold}")
    
    async def process_new_embedding(self, embedding: List[float], people_id: str, 
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new embedding and determine clustering action.
        
        Args:
            embedding: Face embedding vector
            people_id: ID of the embedding in people collection
            metadata: Additional metadata (filename, eventId, orgId, etc.)
            
        Returns:
            Dict containing clustering results
        """
        try:
            filename = metadata.get('filename', 'unknown')
            self.logger.info(f"🧩 CLUSTERING START: {filename} (people_id: {people_id})")
            self.logger.info(f"   Using threshold: {self.similarity_threshold}")
            self.logger.info(f"   Embedding dimensions: {len(embedding)}")
            
            # Search for similar faces
            similar_faces = self._find_similar_faces(embedding)
            
            if similar_faces:
                # Found similar face - link to existing person
                best_match = similar_faces[0]
                similarity_score = best_match['similarity_score']
                existing_face_id = best_match.get('face_id')
                
                self.logger.info(f"🎯 MATCH FOUND: similarity {similarity_score:.6f} vs threshold {self.similarity_threshold}")
                
                if existing_face_id:
                    # Link to existing face
                    success = self.db_service.link_embedding_to_face(people_id, str(existing_face_id))
                    
                    if success:
                        if similarity_score >= 0.95:
                            confidence = "EXCELLENT"
                        elif similarity_score >= 0.90:
                            confidence = "VERY HIGH"
                        elif similarity_score >= 0.85:
                            confidence = "HIGH"
                        else:
                            confidence = "GOOD"
                        
                        self.logger.info(f"✅ LINKED to existing face {existing_face_id}")
                        self.logger.info(f"   Confidence: {confidence} (similarity: {similarity_score:.6f})")
                        self.logger.info(f"   Result: SAME PERSON")
                        
                        return {
                            'action': 'linked_existing',
                            'face_id': str(existing_face_id),
                            'similarity_score': similarity_score,
                            'is_new_face': False
                        }
                    else:
                        self.logger.warning(f"⚠️ Failed to link to existing face {existing_face_id}")
                        # Fall back to creating new face
                        return self._create_new_face(embedding, people_id, metadata)
                else:
                    # Similar embedding exists but no face_id - create face and link both
                    self.logger.info(f"🔗 Similar embedding found but no face_id - grouping embeddings")
                    return self._create_face_from_similar(embedding, people_id, best_match, metadata)
            
            # No similar faces found - create new unique person
            self.logger.info(f"🆕 NO MATCHES found above threshold {self.similarity_threshold}")
            self.logger.info(f"   Result: NEW UNIQUE PERSON")
            return self._create_new_face(embedding, people_id, metadata)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing embedding {people_id}: {str(e)}")
            return {
                'action': 'error',
                'face_id': None,
                'similarity_score': 0.0,
                'is_new_face': False,
                'error': str(e)
            }
    
    def _find_similar_faces(self, embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Find similar faces using vector search.
        
        Args:
            embedding: Target embedding to search for
            
        Returns:
            List of similar faces with similarity scores
        """
        try:
            # Use DB service vector search (which has enhanced logging)
            similar_faces = self.db_service.vector_search_people(
                target_embedding=embedding,
                similarity_threshold=self.similarity_threshold,
                limit=5
            )
            
            return similar_faces
            
        except Exception as e:
            self.logger.error(f"❌ Error in vector search: {str(e)}")
            return []
    
    def _create_new_face(self, embedding: List[float], people_id: str, 
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new unique face entry.
        
        Args:
            embedding: Face embedding
            people_id: ID of the embedding in people collection
            metadata: Additional metadata (filename, eventId, orgId, etc.)
            
        Returns:
            Dict containing creation results
        """
        try:
            filename = metadata.get('filename', 'unknown')
            
            # 🔧 FIXED: Extract eventId and orgId from metadata and pass to create_face
            event_id = metadata.get('eventId')
            org_id = metadata.get('orgId')
            
            person_data = {
                "people_refs": [ObjectId(people_id)],
                "face_count": 1,
                "eventId": event_id,    # 🆕 NEW: Pass eventId context
                "orgId": org_id         # 🆕 NEW: Pass orgId context
            }
            
            face_id = self.db_service.create_face(person_data)
            
            # Update the people document with face_id
            self.db_service.collection.update_one(
                {"_id": ObjectId(people_id)},
                {"$set": {"face_id": ObjectId(face_id)}}
            )
            
            self.logger.info(f"🆕 NEW UNIQUE FACE created: {face_id}")
            self.logger.info(f"   Source: {filename}")
            self.logger.info(f"   EventId: {event_id}")
            self.logger.info(f"   OrgId: {org_id}")
            self.logger.info(f"   People refs: 1")
            self.logger.info(f"   Action: CREATED NEW PERSON")
            
            return {
                'action': 'created_new',
                'face_id': face_id,
                'similarity_score': 0.0,
                'is_new_face': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating new face: {str(e)}")
            raise
    
    def _create_face_from_similar(self, embedding: List[float], people_id: str, 
                                 similar_face: Dict[str, Any], 
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a face when similar embeddings exist but no face_id.
        
        This handles the case where we find similar embeddings that haven't been
        clustered yet.
        """
        try:
            filename = metadata.get('filename', 'unknown')
            similar_people_id = str(similar_face['_id'])
            similarity_score = similar_face['similarity_score']
            similar_filename = similar_face.get('filename', 'unknown')
            
            # 🔧 FIXED: Extract eventId and orgId from metadata and pass to create_face
            event_id = metadata.get('eventId')
            org_id = metadata.get('orgId')
            
            person_data = {
                "people_refs": [ObjectId(people_id), ObjectId(similar_people_id)],
                "face_count": 2,
                "eventId": event_id,    # 🆕 NEW: Pass eventId context
                "orgId": org_id         # 🆕 NEW: Pass orgId context
            }
            
            face_id = self.db_service.create_face(person_data)
            
            # Update both people documents with face_id
            self.db_service.collection.update_one(
                {"_id": ObjectId(people_id)},
                {"$set": {"face_id": ObjectId(face_id)}}
            )
            self.db_service.collection.update_one(
                {"_id": ObjectId(similar_people_id)},
                {"$set": {"face_id": ObjectId(face_id)}}
            )
            
            self.logger.info(f"🔗 GROUPED similar embeddings into new face {face_id}")
            self.logger.info(f"   Similarity: {similarity_score:.6f}")
            self.logger.info(f"   EventId: {event_id}")
            self.logger.info(f"   OrgId: {org_id}")
            self.logger.info(f"   Grouped files: {filename} + {similar_filename}")
            self.logger.info(f"   People refs: 2")
            self.logger.info(f"   Action: GROUPED SIMILAR FACES")
            
            return {
                'action': 'grouped_similar',
                'face_id': face_id,
                'similarity_score': similarity_score,
                'is_new_face': True,
                'grouped_with': similar_people_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error grouping similar faces: {str(e)}")
            raise
    
    def get_clustering_summary(self) -> Dict[str, Any]:
        """
        Get a summary of clustering statistics.
        
        Returns:
            Dict containing clustering statistics
        """
        try:
            stats = self.db_service.get_clustering_stats()
            
            summary = {
                "total_embeddings": stats["total_embeddings"],
                "unique_faces": stats["unique_faces"],
                "clustering_rate": f"{stats['clustering_rate']:.1f}%",
                "unlinked_embeddings": stats["unlinked_embeddings"],
                "similarity_threshold": self.similarity_threshold
            }
            
            # Log clustering summary with System 2 style
            self.logger.info(f"🧩 CLUSTERING SUMMARY:")
            self.logger.info(f"   Total embeddings: {summary['total_embeddings']}")
            self.logger.info(f"   Unique faces: {summary['unique_faces']}")
            self.logger.info(f"   Clustering rate: {summary['clustering_rate']}")
            self.logger.info(f"   Unlinked embeddings: {summary['unlinked_embeddings']}")
            self.logger.info(f"   Threshold used: {summary['similarity_threshold']}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting clustering summary: {str(e)}")
            return {
                "total_embeddings": 0,
                "unique_faces": 0,
                "clustering_rate": "0.0%",
                "unlinked_embeddings": 0,
                "similarity_threshold": self.similarity_threshold
            }
    
    async def process_batch(self, embedding_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple embeddings in batch.
        
        Args:
            embedding_batch: List of embedding data dicts
            
        Returns:
            List of clustering results
        """
        results = []
        
        self.logger.info(f"🧩 BATCH CLUSTERING: Processing {len(embedding_batch)} embeddings")
        
        for i, embedding_data in enumerate(embedding_batch, 1):
            try:
                self.logger.info(f"🧩 Batch item {i}/{len(embedding_batch)}")
                
                result = await self.process_new_embedding(
                    embedding=embedding_data['embedding'],
                    people_id=embedding_data['people_id'],
                    metadata=embedding_data.get('metadata', {})
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"❌ Error in batch processing item {i}: {str(e)}")
                results.append({
                    'action': 'error',
                    'face_id': None,
                    'similarity_score': 0.0,
                    'is_new_face': False,
                    'error': str(e)
                })
        
        # Log batch summary
        successful = len([r for r in results if r['action'] != 'error'])
        new_faces = len([r for r in results if r.get('is_new_face', False)])
        linked_faces = len([r for r in results if r['action'] == 'linked_existing'])
        
        self.logger.info(f"🧩 BATCH COMPLETE: {successful}/{len(embedding_batch)} successful")
        self.logger.info(f"   New faces: {new_faces}")
        self.logger.info(f"   Linked faces: {linked_faces}")
        
        return results
    
    def is_enabled(self) -> bool:
        """Check if face clustering is enabled."""
        return self.settings.ENABLE_FACE_CLUSTERING