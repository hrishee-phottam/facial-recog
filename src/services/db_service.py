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
