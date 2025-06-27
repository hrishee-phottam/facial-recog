import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from src.services.db_service import DBService
from src.config import get_settings

def test_db_insertion():
    """Test inserting a document into MongoDB."""
    print("Testing MongoDB insertion...")
    
    # Get settings
    settings = get_settings()
    
    # Initialize DB service
    db_service = DBService(settings=settings)
    
    # Test document
    test_doc = {
        'filename': 'test_image.jpg',
        'file_path': '/path/to/test_image.jpg',
        'processed_at': datetime.utcnow(),
        'test': True,
        'created_at': datetime.utcnow()
    }
    
    try:
        # Insert document
        print(f"Inserting test document: {test_doc}")
        doc_id = db_service.store_result(test_doc)
        print(f"Successfully inserted document with ID: {doc_id}")
        
        # Verify insertion
        count = db_service.collection.count_documents({'test': True})
        print(f"Found {count} test documents in the collection")
        
        # Clean up
        db_service.collection.delete_one({'_id': doc_id})
        print("Cleaned up test document")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    test_db_insertion()
