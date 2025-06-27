"""
Database service for MongoDB operations.

This module provides the DBService class which handles all database interactions
for the face recognition system.
"""
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
    """Service for interacting with MongoDB."""
    
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
            instance.faces_collection = None  # 🆕 NEW: Faces collection
            
            cls._instance = instance
            # Initialize with settings if provided
            if settings is not None:
                cls._instance.initialize(settings)
                
        return cls._instance
    
    def __init__(self, settings=None):
        """Initialize the database service.
        
        Args:
            settings: Optional settings instance. If not provided, will use global settings.
        """
        # Initialize is handled in __new__ for singleton pattern
        pass
    
    def initialize(self, settings=None):
        """Initialize database connection and collections.
        
        Args:
            settings: Optional settings instance. If not provided, will use global settings.
            
        Raises:
            DBError: If connection or initialization fails.
        """
        if self.initialized:
            return
            
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        try:
            # Log connection attempt
            self.logger.info(f"Connecting to MongoDB at {self.settings.MONGODB_URI}")
            
            # Initialize MongoDB client with connection pooling
            self.client = MongoClient(
                self.settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,         # 10 second connection timeout
                socketTimeoutMS=30000,          # 30 second socket timeout
                maxPoolSize=100,                # Maximum number of connections
                minPoolSize=10,                 # Minimum number of connections
                retryWrites=True,
                retryReads=True
            )
            
            # Test the connection with a ping
            self.logger.debug("Testing MongoDB connection...")
            self.client.admin.command('ping')
            
            # Initialize database and collection
            self.db = self.client[self.settings.MONGODB_DB_NAME]
            self.collection = self.db[self.settings.MONGODB_COLLECTION_NAME]
            
            # 🆕 NEW: Initialize faces collection
            self.faces_collection = self.db[self.settings.FACES_COLLECTION_NAME]
            
            self.logger.debug(f"Using database: {self.settings.MONGODB_DB_NAME}")
            self.logger.debug(f"Using collection: {self.settings.MONGODB_COLLECTION_NAME}")
            self.logger.debug(f"Using faces collection: {self.settings.FACES_COLLECTION_NAME}")
            
            # Create indexes if they don't exist
            self.logger.debug("Ensuring indexes exist...")
            self._ensure_indexes()
            
            self.initialized = True
            self.logger.info("Successfully connected to MongoDB")
            
        except ServerSelectionTimeoutError as e:
            error_msg = "Server selection timeout when connecting to MongoDB"
            self.logger.error(error_msg)
            raise DBError(error_msg) from e
            
        except ConnectionFailure as e:
            error_msg = f"Failed to connect to MongoDB: {str(e)}"
            self.logger.error(error_msg)
            raise DBError(error_msg) from e
            
        except OperationFailure as e:
            error_msg = f"MongoDB operation failed: {str(e)}"
            self.logger.error(error_msg)
            raise DBError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error initializing database: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DBError(error_msg) from e
    
    def _ensure_indexes(self):
        """Ensure required indexes exist on the collection.
        
        Note: TTL index has been removed to ensure documents persist permanently.
        """
        try:
            self.logger.debug("Starting index management...")
            
            # Get existing indexes and log them for debugging
            existing_indexes = self.collection.index_information()
            self.logger.debug(f"Current indexes: {list(existing_indexes.keys())}")
            
            # First, ensure filename index exists
            if 'filename_1' not in existing_indexes:
                self.logger.info("Creating index on 'filename' field")
                self.collection.create_index(
                    [("filename", 1)],
                    name='filename_1'
                )
            else:
                self.logger.debug("'filename_1' index already exists")
            
            # Handle TTL index cleanup if it exists
            for index_name, index_info in existing_indexes.items():
                # Skip the _id_ index
                if index_name == '_id_':
                    continue
                    
                # Check if this is a TTL index
                is_ttl = False
                
                # Handle different index info formats
                if isinstance(index_info, dict):
                    # Check for TTL in the index info
                    if index_info.get('expireAfterSeconds') is not None:
                        is_ttl = True
                    # Check if any of the index fields is a TTL index
                    elif 'key' in index_info:
                        if isinstance(index_info['key'], list):
                            for field, _ in index_info['key']:
                                if field == 'created_at':
                                    is_ttl = True
                                    break
                
                if is_ttl:
                    try:
                        self.logger.info(f"Dropping TTL index: {index_name}")
                        self.collection.drop_index(index_name)
                        self.logger.info(f"Successfully dropped TTL index: {index_name}")
                    except Exception as drop_error:
                        self.logger.warning(f"Failed to drop index {index_name}: {str(drop_error)}")
            
            # 🔧 FIXED: Don't create indexes on fields that don't exist in your schema
            # The faces collection uses your existing schema, no custom indexes needed
            
            self.logger.debug("Index management completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to manage indexes: {str(e)}", exc_info=True)
            raise DBError(f"Index management failed: {str(e)}") from e
    
    def store_result(self, data: Dict[str, Any]) -> str:
        """Store face recognition result in the database.
        
        Args:
            data: Dictionary containing the face recognition result.
            
        Returns:
            str: The ID of the inserted document.
            
        Raises:
            DBError: If the operation fails.
        """
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
        return {
            'total_documents': self.collection.count_documents({}),
            'last_processed': self.collection.find_one(
                {},
                sort=[('processed_at', -1)]
            )
        }
    
    # 🆕 NEW: Vector Search Methods
    
    def vector_search_people(self, target_embedding: List[float], 
                            similarity_threshold: float = 0.85, 
                            limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar face embeddings using MongoDB Vector Search.
        
        Args:
            target_embedding: The embedding to search for
            similarity_threshold: Minimum similarity score to return
            limit: Maximum number of results
            
        Returns:
            List of similar embeddings with similarity scores
        """
        if not self.initialized:
            self.initialize()
            
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.settings.VECTOR_INDEX_NAME,  # "face_embedding_index"
                        "path": "embedding",
                        "queryVector": target_embedding,
                        "numCandidates": 200,
                        "limit": limit * 2  # Get more to filter by threshold
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
                        "face_id": 1  # Include if exists
                    }
                },
                {
                    "$match": {
                        "similarity_score": {"$gte": similarity_threshold}
                    }
                },
                {
                    "$sort": {"similarity_score": -1}
                },
                {
                    "$limit": limit
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            self.logger.debug(f"Vector search found {len(results)} similar faces above threshold {similarity_threshold}")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            return []
    
    def find_embedding_by_id(self, people_id: str) -> Optional[Dict[str, Any]]:
        """Find embedding document by ID."""
        if not self.initialized:
            self.initialize()
            
        try:
            return self.collection.find_one({"_id": ObjectId(people_id)})
        except Exception as e:
            self.logger.error(f"Error finding embedding {people_id}: {str(e)}")
            return None
    
    # 🆕 NEW: Faces Collection Methods
    
    def create_face(self, person_data: Dict[str, Any]) -> str:
        """
        Create a new unique face entry matching existing faces collection schema.
        
        Args:
            person_data: Data for the new face
            
        Returns:
            str: The ID of the created face
        """
        if not self.initialized:
            self.initialize()
            
        try:
            # 🔧 FIXED: Match your exact faces collection structure
            face_doc = {
                # Fields we DON'T have from new system - set to null
                "eventId": None,
                "orgId": None, 
                "thumbnail": None,
                "tScore": None,
                
                # Fields we CAN provide
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                
                # 🆕 NEW: Additional fields for our clustering (optional)
                "people_refs": person_data.get("people_refs", []),
                "face_count": person_data.get("face_count", 1)
            }
            
            result = self.faces_collection.insert_one(face_doc)
            face_id = str(result.inserted_id)
            
            self.logger.info(f"Created new face: {face_id}")
            return face_id
            
        except Exception as e:
            self.logger.error(f"Error creating face: {str(e)}")
            raise DBError(f"Failed to create face: {str(e)}") from e
    
    def link_embedding_to_face(self, people_id: str, face_id: str) -> bool:
        """
        Link an embedding to an existing face.
        
        Args:
            people_id: ID of the embedding in people collection
            face_id: ID of the face to link to
            
        Returns:
            bool: True if successful
        """
        if not self.initialized:
            self.initialize()
            
        try:
            # Update people document with face_id
            people_result = self.collection.update_one(
                {"_id": ObjectId(people_id)},
                {"$set": {"face_id": ObjectId(face_id)}}
            )
            
            # 🔧 FIXED: Update faces document - only modify fields we control
            faces_result = self.faces_collection.update_one(
                {"_id": ObjectId(face_id)},
                {
                    "$addToSet": {"people_refs": ObjectId(people_id)},
                    "$inc": {"face_count": 1},
                    "$set": {"updatedAt": datetime.utcnow()}  # Use your exact field name
                }
            )
            
            success = people_result.modified_count > 0 and faces_result.modified_count > 0
            if success:
                self.logger.debug(f"Linked embedding {people_id} to face {face_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error linking embedding to face: {str(e)}")
            return False
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face document by ID."""
        if not self.initialized:
            self.initialize()
            
        try:
            return self.faces_collection.find_one({"_id": ObjectId(face_id)})
        except Exception as e:
            self.logger.error(f"Error getting face {face_id}: {str(e)}")
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
            self.logger.error(f"Error getting clustering stats: {str(e)}")
            return {
                "total_embeddings": 0,
                "linked_embeddings": 0,
                "unlinked_embeddings": 0,
                "unique_faces": 0,
                "clustering_rate": 0
            }