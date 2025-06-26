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
            self.logger.debug(f"Using database: {self.settings.MONGODB_DB_NAME}")
            self.logger.debug(f"Using collection: {self.settings.MONGODB_COLLECTION_NAME}")
            
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
        """Ensure required indexes exist on the collection."""
        try:
            # Create index on filename for faster lookups
            self.collection.create_index([("filename", 1)])
            # Create TTL index for documents that should expire
            self.collection.create_index(
                [("created_at", 1)],
                expireAfterSeconds=0  # Will be set by the document's expireAt field
            )
        except OperationFailure as e:
            self.logger.warning(f"Failed to create indexes: {str(e)}")
    
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
            raise Exception(f"Failed to store data in MongoDB: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        return {
            'total_documents': self.collection.count_documents({}),
            'last_processed': self.collection.find_one(
                {},
                sort=[('processed_at', -1)]
            )
        }
        
    def find_similar_faces(
        self, 
        embedding: List[float], 
        max_results: int = 5, 
        min_score: float = 0.7,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Find similar faces using MongoDB's vector search
        
        Args:
            embedding: The face embedding vector to search with
            max_results: Maximum number of results to return
            min_score: Minimum similarity score (0-1) for results
            include_metadata: Whether to include full document or just similarity score
            
        Returns:
            List of matching documents with similarity scores
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "face_embeddings",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": max_results * 2,  # Get more to filter by score
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "filename": 1,
                    "box": 1,
                    "processed_at": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {"$match": {"score": {"$gte": min_score}}},
            {"$limit": max_results}
        ]
        
        if include_metadata:
            pipeline.append({
                "$lookup": {
                    "from": self.collection.name,
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "metadata"
                }
            })
            pipeline.append({"$unwind": "$metadata"})
        
        try:
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logging.error(f"Vector search failed: {str(e)}")
            return []
