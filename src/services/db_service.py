from pymongo import MongoClient
from typing import Dict, Any, List, Optional
import os
import logging

class DBService:
    """Singleton database service class"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBService, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize database connection using environment variables"""
        # Required MongoDB configuration
        required_vars = [
            'MONGODB_URI',
            'MONGODB_DB_NAME',
            'MONGODB_COLLECTION_NAME'
        ]
        
        # Check for missing required environment variables
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required database configuration: {', '.join(missing_vars)}")
        
        # Get MongoDB configuration from environment variables
        mongo_uri = os.environ['MONGODB_URI']
        db_name = os.environ['MONGODB_DB_NAME']
        collection_name = os.environ['MONGODB_COLLECTION_NAME']
        
        # Initialize MongoDB client
        try:
            self.client = MongoClient(mongo_uri)
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

    def store_result(self, data: Dict[str, Any]) -> str:
        """Store face recognition result in database"""
        try:
            result = self.collection.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
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
