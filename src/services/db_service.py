import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    OperationFailure,
    PyMongoError
)
from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId

from src.config import get_settings


class DBError(Exception):
    """Custom exception for database-related errors."""
    pass


class DBService:
    """Enhanced database service for MongoDB with production media operations."""
    
    _instance = None
    
    def __new__(cls, settings=None):
        """Implement singleton pattern."""
        if cls._instance is None:
            instance = super(DBService, cls).__new__(cls)
            # Initialize the instance attributes here to ensure they exist
            instance.initialized = False
            instance.settings = None
            instance.logger = None
            instance.client = None
            instance.db = None
            instance.collection = None
            instance.faces_collection = None
            instance.media_collection = None
            instance.events_collection = None
            
            cls._instance = instance
            # Initialize with settings if provided
            if settings is not None:
                cls._instance.initialize(settings)
                
        return cls._instance
    
    def __init__(self, settings=None):
        """Initialize the database service."""
        # Initialize is handled in __new__ for singleton pattern
        pass
    
    def initialize(self, settings=None):
        """Initialize database connection and collections."""
        if self.initialized:
            return
            
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info(f"🔗 Connecting to MongoDB...")
            
            # Initialize MongoDB client with connection pooling
            self.client = MongoClient(
                self.settings.MONGODB_URI,
                serverSelectionTimeoutMS=self.settings.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
                connectTimeoutMS=self.settings.MONGODB_CONNECT_TIMEOUT_MS,
                socketTimeoutMS=30000,
                maxPoolSize=100,
                minPoolSize=10,
                retryWrites=True,
                retryReads=True
            )
            
            # Test the connection with a ping
            self.client.admin.command('ping')
            
            # Initialize database and collections
            self.db = self.client[self.settings.MONGODB_DB_NAME]
            self.collection = self.db[self.settings.MONGODB_COLLECTION_NAME]
            
            # Initialize all collections
            self.faces_collection = self.db[self.settings.FACES_COLLECTION_NAME]
            self.media_collection = self.db["media"]
            self.events_collection = self.db["events"]
            
            # Create indexes if they don't exist
            self._ensure_indexes()
            
            self.initialized = True
            self.logger.info(f"✅ MongoDB connected: {self.settings.MONGODB_DB_NAME}")
            
        except ServerSelectionTimeoutError as e:
            error_msg = "❌ MongoDB server selection timeout"
            self.logger.error(error_msg)
            raise DBError(error_msg) from e
            
        except ConnectionFailure as e:
            error_msg = f"❌ MongoDB connection failed: {str(e)}"
            self.logger.error(error_msg)
            raise DBError(error_msg) from e
            
        except OperationFailure as e:
            error_msg = f"❌ MongoDB operation failed: {str(e)}"
            self.logger.error(error_msg)
            raise DBError(error_msg) from e
            
        except Exception as e:
            error_msg = f"❌ Unexpected database error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DBError(error_msg) from e
    
    def _ensure_indexes(self):
        """Ensure required indexes exist on the collections."""
        try:
            existing_indexes = self.collection.index_information()
            
            # Ensure filename index exists on people collection
            if 'filename_1' not in existing_indexes:
                self.collection.create_index([("filename", 1)], name='filename_1')
            
            # Ensure mediaId index exists on people collection
            if 'mediaId_1' not in existing_indexes:
                self.collection.create_index([("mediaId", 1)], name='mediaId_1')
            
            # Ensure indexes on media collection
            media_indexes = self.media_collection.index_information()
            if 'path_1' not in media_indexes:
                self.media_collection.create_index([("path", 1)], name='path_1')
            
            # Ensure indexes on events collection
            events_indexes = self.events_collection.index_information()
            if 'faceGrouping_1' not in events_indexes:
                self.events_collection.create_index([("faceGrouping", 1)], name='faceGrouping_1')
            
        except Exception as e:
            self.logger.error(f"❌ Index management failed: {str(e)}", exc_info=True)
            raise DBError(f"Index management failed: {str(e)}") from e
    
    # ==================== EXISTING SYSTEM 2 METHODS (UNCHANGED) ====================
    
    def store_result(self, data: Dict[str, Any]) -> str:
        """Store face recognition result in the database."""
        if not self.initialized:
            self.initialize()
            
        try:
            # Add timestamp
            data['created_at'] = datetime.utcnow()
            
            # Insert the document
            result = self.collection.insert_one(data)
            return str(result.inserted_id)
            
        except OperationFailure as e:
            raise DBError(f"Database operation failed: {str(e)}") from e
        except PyMongoError as e:
            raise DBError(f"MongoDB error: {str(e)}") from e
        except Exception as e:
            raise DBError(f"Failed to store result: {str(e)}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        if not self.initialized:
            self.initialize()
            
        try:
            return {
                'total_documents': self.collection.count_documents({}),
                'last_processed': self.collection.find_one(
                    {},
                    sort=[('processed_at', -1)]
                )
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting stats: {str(e)}")
            return {'total_documents': 0, 'last_processed': None}
    
    # ==================== ENHANCED VECTOR SEARCH WITH CLEAN STRUCTURED LOGGING ====================
    
    def _get_closeness_level(self, gap: float) -> str:
        """Get closeness level indicator based on gap"""
        if gap < 0.05:
            return "🔥 VERY CLOSE "
        elif gap < 0.10:
            return "⚠️  CLOSE      "
        elif gap < 0.20:
            return "📊 MODERATE    "
        else:
            return "📉 DISTANT     "
    
    def _get_confidence_level(self, similarity: float) -> str:
        """Get confidence level for matches above threshold"""
        if similarity >= 0.95:
            return "EXCELLENT MATCH"
        elif similarity >= 0.90:
            return "VERY HIGH MATCH"
        elif similarity >= 0.85:
            return "HIGH MATCH     "
        else:
            return "GOOD MATCH     "
    
    def vector_search_people(self, target_embedding: List[float], 
                            similarity_threshold: float = 0.85, 
                            limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar face embeddings using MongoDB Vector Search with enhanced structured logging."""
        if not self.initialized:
            self.initialize()
            
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.settings.VECTOR_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": target_embedding,
                        "numCandidates": 200,
                        "limit": limit * 3  # Get more to show near-misses
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "filename": 1,
                        "file_path": 1,
                        "embedding": 1,
                        "similarity_score": 1,
                        "face_id": 1
                    }
                },
                {
                    "$sort": {"similarity_score": -1}
                }
            ]
            
            all_results = list(self.collection.aggregate(pipeline))
            
            # Separate matches above and below threshold
            matches_above_threshold = [r for r in all_results if r['similarity_score'] >= similarity_threshold]
            near_misses = [r for r in all_results if r['similarity_score'] < similarity_threshold]
            
            # Enhanced structured logging
            self.logger.info("   🔍 VECTOR SEARCH RESULTS:")
            
            # Matches above threshold section
            self.logger.info("   ┌── MATCHES ABOVE THRESHOLD (≥{}) ──────────────────────────────────────".format(similarity_threshold))
            if matches_above_threshold:
                for i, result in enumerate(matches_above_threshold, 1):
                    score = result['similarity_score']
                    face_id = str(result.get('face_id', 'none'))[-8:] if result.get('face_id') else 'none'
                    filename = result.get('filename', 'unknown')
                    confidence = self._get_confidence_level(score)
                    self.logger.info(f"   │   🎯 {confidence} │ {score:.3f} │ Face: {face_id} │ {filename} │ SAME PERSON")
            else:
                self.logger.info("   │   ❌ No matches found above threshold")
            self.logger.info("   └─────────────────────────────────────────────────────────────────────────")
            
            # Near misses analysis section (only if we found matches above threshold OR if no matches)
            if near_misses:
                if matches_above_threshold:
                    section_title = "ADDITIONAL CONTEXT (Below threshold)"
                else:
                    section_title = "SIMILARITY ANALYSIS"
                
                self.logger.info(f"   ┌── {section_title} ─────────────────────────────────")
                
                # Show top 5 near misses
                for i, miss in enumerate(near_misses[:5], 1):
                    score = miss['similarity_score']
                    face_id_short = str(miss.get('face_id', 'none'))[-8:] if miss.get('face_id') else 'none'
                    gap = similarity_threshold - score
                    filename = miss.get('filename', 'unknown')
                    closeness = self._get_closeness_level(gap)
                    
                    self.logger.info(f"   │   {closeness} │ {score:.3f} (gap: -{gap:.3f}) │ Face: {face_id_short} │ {filename}")
                
                self.logger.info("   └─────────────────────────────────────────────────────────────────────────")
                
                # Add decision context only if no matches above threshold
                if not matches_above_threshold:
                    closest_score = near_misses[0]['similarity_score']
                    gap = similarity_threshold - closest_score
                    
                    if gap < 0.05:
                        context = f"VERY CLOSE to existing face (gap: {gap:.3f}) - consider lowering threshold"
                    elif gap < 0.10:
                        context = f"CLOSE to existing face (gap: {gap:.3f}) - possible same person"
                    elif gap < 0.20:
                        context = f"MODERATE similarity (gap: {gap:.3f}) - likely different person"
                    else:
                        context = f"CLEARLY different (gap: {gap:.3f}) - definitely new person"
                    
                    self.logger.info(f"   🎯 NEW FACE CONTEXT: {context}")
                    self.logger.info(f"   🎯 DECISION CONTEXT: Closest existing face scored {closest_score:.6f}, needed {similarity_threshold:.2f}")
            
            return matches_above_threshold
            
        except Exception as e:
            self.logger.error(f"❌ Vector search failed: {str(e)}")
            return []
    
    def find_embedding_by_id(self, people_id: str) -> Optional[Dict[str, Any]]:
        """Find embedding document by ID."""
        if not self.initialized:
            self.initialize()
            
        try:
            return self.collection.find_one({"_id": ObjectId(people_id)})
        except Exception as e:
            self.logger.error(f"❌ Error finding embedding {people_id}: {str(e)}")
            return None
    
    # ==================== ENHANCED SYSTEM 2 FACES COLLECTION METHODS ====================
    
    def create_face(self, person_data: Dict[str, Any]) -> str:
        """Create a new unique face entry matching existing faces collection schema."""
        if not self.initialized:
            self.initialize()
            
        try:
            # Use eventId and orgId from person_data
            event_id = person_data.get("eventId")
            org_id = person_data.get("orgId")
            
            # Convert string IDs to ObjectIds if needed
            if event_id and isinstance(event_id, str):
                try:
                    event_id = ObjectId(event_id)
                except:
                    event_id = None
            
            if org_id and isinstance(org_id, str):
                try:
                    org_id = ObjectId(org_id)
                except:
                    org_id = None
            
            face_doc = {
                "eventId": event_id,
                "orgId": org_id,
                "thumbnail": None,
                "tScore": None,
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "people_refs": person_data.get("people_refs", []),
                "face_count": person_data.get("face_count", 1)
            }
            
            result = self.faces_collection.insert_one(face_doc)
            face_id = str(result.inserted_id)
            
            # Generate thumbnail path if we have eventId
            if event_id:
                thumbnail_path = f"facetn/{str(event_id)}/{face_id}.jpg"
                # Update the face document with thumbnail path
                self.faces_collection.update_one(
                    {"_id": ObjectId(face_id)},
                    {"$set": {"thumbnail": thumbnail_path}}
                )
            
            return face_id
            
        except Exception as e:
            self.logger.error(f"❌ Error creating face: {str(e)}")
            raise DBError(f"Failed to create face: {str(e)}") from e
    
    def link_embedding_to_face(self, people_id: str, face_id: str) -> bool:
        """Link an embedding to an existing face."""
        if not self.initialized:
            self.initialize()
            
        try:
            # Update people document with face_id
            people_result = self.collection.update_one(
                {"_id": ObjectId(people_id)},
                {"$set": {"face_id": ObjectId(face_id)}}
            )
            
            # Update faces document
            faces_result = self.faces_collection.update_one(
                {"_id": ObjectId(face_id)},
                {
                    "$addToSet": {"people_refs": ObjectId(people_id)},
                    "$inc": {"face_count": 1},
                    "$set": {"updatedAt": datetime.utcnow()}
                }
            )
            
            success = people_result.modified_count > 0 and faces_result.modified_count > 0
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error linking embedding to face: {str(e)}")
            return False
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face document by ID."""
        if not self.initialized:
            self.initialize()
            
        try:
            return self.faces_collection.find_one({"_id": ObjectId(face_id)})
        except Exception as e:
            self.logger.error(f"❌ Error getting face {face_id}: {str(e)}")
            return None
    
    def get_clustering_stats(self) -> Dict[str, Any]:
        """Get statistics about face clustering."""
        if not self.initialized:
            self.initialize()
            
        try:
            total_embeddings = self.collection.count_documents({})
            linked_embeddings = self.collection.count_documents({"face_id": {"$exists": True}})
            total_faces = self.faces_collection.count_documents({})
            
            return {
                "total_embeddings": total_embeddings,
                "linked_embeddings": linked_embeddings,
                "unlinked_embeddings": total_embeddings - linked_embeddings,
                "unique_faces": total_faces,
                "clustering_rate": (linked_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting clustering stats: {str(e)}")
            return {
                "total_embeddings": 0,
                "linked_embeddings": 0,
                "unlinked_embeddings": 0,
                "unique_faces": 0,
                "clustering_rate": 0
            }
    
    # ==================== NEW PRODUCTION METHODS FOR SQS PROCESSING ====================
    
    async def check_event_ai_enabled(self, event_id: str) -> bool:
        """Check if AI face recognition is enabled for an event."""
        if not self.initialized:
            self.initialize()
            
        try:
            event = None
            
            # Try as ObjectId if it looks like one
            try:
                if len(event_id) == 24:
                    object_id = ObjectId(event_id)
                    event = self.events_collection.find_one(
                        {"_id": object_id},
                        {"faceGrouping": 1, "name": 1}
                    )
            except:
                pass
            
            # If not found, try as string
            if not event:
                event = self.events_collection.find_one(
                    {"_id": event_id},
                    {"faceGrouping": 1, "name": 1}
                )
            
            if not event:
                self.logger.warning(f"⚠️ Event {event_id} not found")
                return False
            
            ai_enabled = event.get('faceGrouping', False)
            
            return ai_enabled
            
        except Exception as e:
            self.logger.error(f"❌ Error checking AI enabled for event {event_id}: {str(e)}")
            return False
    
    async def get_media_by_id(self, media_id: str) -> Optional[Dict[str, Any]]:
        """Get media document by ID."""
        if not self.initialized:
            self.initialize()
            
        try:
            media = self.media_collection.find_one({"_id": ObjectId(media_id)})
            return media
        except Exception as e:
            self.logger.error(f"❌ Error getting media {media_id}: {str(e)}")
            return None
    
    async def update_media_status(self, media_id: str, status: str) -> bool:
        """Update media processing status."""
        if not self.initialized:
            self.initialize()
            
        try:
            result = self.media_collection.update_one(
                {"_id": ObjectId(media_id)},
                {
                    "$set": {
                        "embedStatus": status,
                        "updatedAt": datetime.utcnow()
                    }
                }
            )
            
            success = result.modified_count > 0
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error updating media status {media_id}: {str(e)}")
            return False
    
    async def update_media_with_faces(self, media_id: str, face_ids: List[str]) -> bool:
        """Update media document with detected face references."""
        if not self.initialized:
            self.initialize()
            
        try:
            # Convert string IDs to ObjectIds
            face_object_ids = [ObjectId(fid) for fid in face_ids]
            
            result = self.media_collection.update_one(
                {"_id": ObjectId(media_id)},
                {
                    "$set": {
                        "faces": face_object_ids,
                        "updatedAt": datetime.utcnow()
                    }
                }
            )
            
            success = result.modified_count > 0
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error updating media with faces {media_id}: {str(e)}")
            return False
    
    def store_result_with_media(self, data: Dict[str, Any], media_id: str) -> str:
        """Store face recognition result with media reference."""
        if not self.initialized:
            self.initialize()
            
        try:
            # Add media reference and timestamp
            data['mediaId'] = ObjectId(media_id)
            data['created_at'] = datetime.utcnow()
            
            # Insert the document
            result = self.collection.insert_one(data)
            people_id = str(result.inserted_id)
            
            return people_id
            
        except OperationFailure as e:
            raise DBError(f"Database operation failed: {str(e)}") from e
        except PyMongoError as e:
            raise DBError(f"MongoDB error: {str(e)}") from e
        except Exception as e:
            raise DBError(f"Failed to store result with media: {str(e)}") from e
    
    def get_media_by_path(self, s3_path: str) -> Optional[Dict[str, Any]]:
        """Get media document by S3 path."""
        if not self.initialized:
            self.initialize()
            
        try:
            media = self.media_collection.find_one({"path": s3_path})
            return media
        except Exception as e:
            self.logger.error(f"❌ Error getting media by path {s3_path}: {str(e)}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            self.initialized = False
            self.logger.info("🔗 MongoDB connection closed")


# Global database service instance
_db_service = None


def get_database_service() -> DBService:
    """Get initialized database service instance."""
    global _db_service
    if _db_service is None or not _db_service.initialized:
        _db_service = DBService()
        _db_service.initialize()
    return _db_service