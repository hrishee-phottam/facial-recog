import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import time
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
    
    def log_clustering_header(self, face_index: int, total_faces: int, people_id: str, threshold: float):
        """Log face clustering section header"""
        self.logger.info(f"🧩 FACE CLUSTERING: Face {face_index}/{total_faces} ────────────────────────────────────────────────")
        self.logger.info(f"   📋 Input: people_id={people_id} | dims=512 | threshold={threshold}")
        self.logger.info("")
    
    def log_clustering_decision(self, action: str, face_id: str = None, similarity: float = 0.0, 
                              closest_similarity: float = 0.0, confidence: str = ""):
        """Log clustering decision with timing"""
        if action == 'linked_existing':
            self.logger.info(f"   🎯 DECISION: LINK TO EXISTING PERSON (confidence: {confidence} - {similarity:.3f}) ✅")
        elif action == 'created_new':
            self.logger.info(f"   🎯 DECISION: CREATE NEW PERSON → Face ID: {face_id} ✅")
        elif action == 'grouped_similar':
            self.logger.info(f"   🎯 DECISION: GROUP SIMILAR FACES → Face ID: {face_id} ✅")
    
    def log_face_time(self, search_time: float, db_time: float, thumbnail_time: float):
        """Log individual face processing time breakdown"""
        total_time = search_time + db_time + thumbnail_time
        self.logger.info(f"   ⏱️  Face Time: {total_time:.1f}s (Search: {search_time:.1f}s | DB Ops: {db_time:.1f}s | Thumbnail: {thumbnail_time:.1f}s)")
        self.logger.info("")
    
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
        search_start = time.time()
        
        try:
            filename = metadata.get('filename', 'unknown')
            
            # Search for similar faces
            similar_faces = self._find_similar_faces(embedding)
            search_time = time.time() - search_start
            
            db_start = time.time()
            
            if similar_faces:
                # Found similar face - link to existing person
                best_match = similar_faces[0]
                similarity_score = best_match['similarity_score']
                existing_face_id = best_match.get('face_id')
                
                if existing_face_id:
                    # Link to existing face
                    success = self.db_service.link_embedding_to_face(people_id, str(existing_face_id))
                    
                    if success:
                        confidence = self._get_confidence_level(similarity_score)
                        
                        db_time = time.time() - db_start
                        thumbnail_time = 0  # Thumbnail handled in processor
                        
                        # Log decision
                        self.log_clustering_decision('linked_existing', str(existing_face_id), 
                                                   similarity_score, 0, confidence)
                        
                        return {
                            'action': 'linked_existing',
                            'face_id': str(existing_face_id),
                            'similarity_score': similarity_score,
                            'is_new_face': False,
                            'timing': {
                                'search_time': search_time,
                                'db_time': db_time,
                                'thumbnail_time': thumbnail_time
                            }
                        }
                    else:
                        # Fall back to creating new face
                        return self._create_new_face(embedding, people_id, metadata, search_time)
                else:
                    # Similar embedding exists but no face_id - create face and link both
                    return self._create_face_from_similar(embedding, people_id, best_match, metadata, search_time)
            
            # No similar faces found - create new unique person
            return self._create_new_face(embedding, people_id, metadata, search_time)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing embedding {people_id}: {str(e)}")
            return {
                'action': 'error',
                'face_id': None,
                'similarity_score': 0.0,
                'is_new_face': False,
                'error': str(e)
            }
    
    def _get_confidence_level(self, similarity: float) -> str:
        """Get confidence level description"""
        if similarity >= 0.95:
            return "EXCELLENT"
        elif similarity >= 0.90:
            return "VERY HIGH"
        elif similarity >= 0.85:
            return "HIGH"
        else:
            return "GOOD"
    
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
                        metadata: Dict[str, Any], search_time: float) -> Dict[str, Any]:
        """
        Create a new unique face entry.
        
        Args:
            embedding: Face embedding
            people_id: ID of the embedding in people collection
            metadata: Additional metadata (filename, eventId, orgId, etc.)
            search_time: Time spent on vector search
            
        Returns:
            Dict containing creation results
        """
        try:
            db_start = time.time()
            
            filename = metadata.get('filename', 'unknown')
            event_id = metadata.get('eventId')
            org_id = metadata.get('orgId')
            
            person_data = {
                "people_refs": [ObjectId(people_id)],
                "face_count": 1,
                "eventId": event_id,
                "orgId": org_id
            }
            
            face_id = self.db_service.create_face(person_data)
            
            # Update the people document with face_id
            self.db_service.collection.update_one(
                {"_id": ObjectId(people_id)},
                {"$set": {"face_id": ObjectId(face_id)}}
            )
            
            db_time = time.time() - db_start
            thumbnail_time = 0  # Thumbnail handled in processor
            
            # Log decision
            self.log_clustering_decision('created_new', face_id)
            
            # Find closest similarity for context
            closest_similarity = 0.0
            try:
                # Get the closest match from the search for logging context
                all_results = self.db_service.vector_search_people(
                    target_embedding=embedding,
                    similarity_threshold=0.0,  # Get all results
                    limit=1
                )
                if all_results:
                    closest_similarity = all_results[0]['similarity_score']
            except:
                pass
            
            return {
                'action': 'created_new',
                'face_id': face_id,
                'similarity_score': 0.0,
                'is_new_face': True,
                'closest_similarity': closest_similarity,
                'timing': {
                    'search_time': search_time,
                    'db_time': db_time,
                    'thumbnail_time': thumbnail_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating new face: {str(e)}")
            raise
    
    def _create_face_from_similar(self, embedding: List[float], people_id: str, 
                                 similar_face: Dict[str, Any], 
                                 metadata: Dict[str, Any], search_time: float) -> Dict[str, Any]:
        """
        Create a face when similar embeddings exist but no face_id.
        
        This handles the case where we find similar embeddings that haven't been
        clustered yet.
        """
        try:
            db_start = time.time()
            
            filename = metadata.get('filename', 'unknown')
            similar_people_id = str(similar_face['_id'])
            similarity_score = similar_face['similarity_score']
            similar_filename = similar_face.get('filename', 'unknown')
            
            event_id = metadata.get('eventId')
            org_id = metadata.get('orgId')
            
            person_data = {
                "people_refs": [ObjectId(people_id), ObjectId(similar_people_id)],
                "face_count": 2,
                "eventId": event_id,
                "orgId": org_id
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
            
            db_time = time.time() - db_start
            thumbnail_time = 0  # Thumbnail handled in processor
            
            # Log decision
            self.log_clustering_decision('grouped_similar', face_id, similarity_score)
            
            return {
                'action': 'grouped_similar',
                'face_id': face_id,
                'similarity_score': similarity_score,
                'is_new_face': True,
                'grouped_with': similar_people_id,
                'timing': {
                    'search_time': search_time,
                    'db_time': db_time,
                    'thumbnail_time': thumbnail_time
                }
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
            
            # Log clustering summary
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