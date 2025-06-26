from pymongo import MongoClient
from typing import Dict, Any
import os

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
        # Get MongoDB configuration from environment variables
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        db_name = os.getenv('MONGODB_DB_NAME', 'phottam')
        collection_name = os.getenv('MONGODB_COLLECTION_NAME', 'people')
        
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

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
