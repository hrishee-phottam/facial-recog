"""
Tests for the DBService class.
"""
import os
import sys
import tempfile
from unittest import TestCase, mock
from unittest.mock import MagicMock, patch, call

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError

from src.services.db_service import DBService, DBError
from src.config.settings import Settings


class TestDBService(TestCase):
    """Test cases for the DBService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test settings object
        self.settings = Settings(
            MONGODB_URI="mongodb://localhost:27017",
            MONGODB_DB_NAME="test_db",
            MONGODB_COLLECTION_NAME="test_collection",
            MONGODB_CONNECT_TIMEOUT_MS=1000,  # Shorter timeout for tests
            MONGODB_SERVER_SELECTION_TIMEOUT_MS=1000,
            LOG_LEVEL="WARNING"
        )
        
        # Patch the MongoClient to use a mock
        self.client_patcher = patch('pymongo.MongoClient')
        self.mock_mongo_client = self.client_patcher.start()
        
        # Set up the mock client and collections
        self.mock_client = MagicMock(spec=MongoClient)
        self.mock_db = MagicMock()
        self.mock_collection = MagicMock()
        
        self.mock_client.__getitem__.return_value = self.mock_db
        self.mock_db.__getitem__.return_value = self.mock_collection
        
        self.mock_mongo_client.return_value = self.mock_client
        
        # Initialize the DB service
        self.db_service = DBService(settings=self.settings)
    
    def tearDown(self):
        """Clean up after tests."""
        self.client_patcher.stop()
    
    def test_initialization(self):
        """Test that the DB service initializes correctly."""
        # Check that the client was created with the correct parameters
        self.mock_mongo_client.assert_called_once_with(
            self.settings.MONGODB_URI,
            connectTimeoutMS=self.settings.MONGODB_CONNECT_TIMEOUT_MS,
            serverSelectionTimeoutMS=self.settings.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
            retryWrites=True,
            w='majority'
        )
        
        # Check that the correct database and collection are used
        self.mock_client.__getitem__.assert_called_once_with(self.settings.MONGODB_DB_NAME)
        self.mock_db.__getitem__.assert_called_once_with(self.settings.MONGODB_COLLECTION_NAME)
        
        # Check that indexes were created
        self.mock_collection.create_indexes.assert_called_once()
    
    def test_initialization_connection_error(self):
        """Test that a connection error during initialization is handled."""
        # Make the client raise a connection error
        self.mock_mongo_client.side_effect = ConnectionFailure("Could not connect to MongoDB")
        
        with self.assertRaises(DBError) as context:
            DBService(settings=self.settings)
        
        self.assertIn("Could not connect to MongoDB", str(context.exception))
    
    def test_save_result_success(self):
        """Test saving a result successfully."""
        # Set up test data
        file_path = "/path/to/image.jpg"
        result = {"result": [{"face_id": 1, "confidence": 0.95}]}
        
        # Mock the insert_one method
        insert_result = MagicMock()
        insert_result.inserted_id = "test_id_123"
        self.mock_collection.insert_one.return_value = insert_result
        
        # Call the method
        doc_id = self.db_service.save_result(
            file_path=file_path,
            result=result,
            success=True,
            error_message=None
        )
        
        # Check the result
        self.assertEqual(doc_id, "test_id_123")
        
        # Check that insert_one was called with the correct document
        self.mock_collection.insert_one.assert_called_once()
        doc = self.mock_collection.insert_one.call_args[0][0]
        
        self.assertEqual(doc['file_path'], file_path)
        self.assertEqual(doc['result'], result)
        self.assertTrue(doc['success'])
        self.assertNotIn('error_message', doc)
        self.assertIn('timestamp', doc)
    
    def test_save_result_with_error(self):
        """Test saving a result with an error message."""
        # Set up test data
        file_path = "/path/to/error.jpg"
        error_message = "Face not found"
        
        # Mock the insert_one method
        insert_result = MagicMock()
        insert_result.inserted_id = "test_error_id"
        self.mock_collection.insert_one.return_value = insert_result
        
        # Call the method
        doc_id = self.db_service.save_result(
            file_path=file_path,
            result=None,
            success=False,
            error_message=error_message
        )
        
        # Check the result
        self.assertEqual(doc_id, "test_error_id")
        
        # Check that insert_one was called with the correct document
        doc = self.mock_collection.insert_one.call_args[0][0]
        self.assertEqual(doc['file_path'], file_path)
        self.assertNotIn('result', doc)
        self.assertFalse(doc['success'])
        self.assertEqual(doc['error_message'], error_message)
    
    def test_save_result_operation_failure(self):
        """Test handling of operation failures during save."""
        # Set up test data
        file_path = "/path/to/image.jpg"
        
        # Make insert_one raise an operation failure
        self.mock_collection.insert_one.side_effect = OperationFailure("Write error")
        
        # Call the method and check that it raises DBError
        with self.assertRaises(DBError) as context:
            self.db_service.save_result(
                file_path=file_path,
                result={"test": "data"},
                success=True
            )
        
        self.assertIn("Failed to save result to database", str(context.exception))
    
    def test_find_results(self):
        """Test finding results in the database."""
        # Set up test data
        test_results = [
            {"file_path": "/path/to/image1.jpg", "success": True},
            {"file_path": "/path/to/image2.jpg", "success": False, "error_message": "No face found"}
        ]
        
        # Mock the find method
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = test_results
        self.mock_collection.find.return_value = mock_cursor
        
        # Call the method
        query = {"success": True}
        results = list(self.db_service.find_results(query))
        
        # Check the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results, test_results)
        
        # Check that find was called with the correct query
        self.mock_collection.find.assert_called_once_with(query)
    
    def test_find_results_error(self):
        """Test error handling in find_results."""
        # Make find raise an operation failure
        self.mock_collection.find.side_effect = OperationFailure("Query failed")
        
        # Call the method and check that it raises DBError
        with self.assertRaises(DBError) as context:
            list(self.db_service.find_results({}))
        
        self.assertIn("Failed to query database", str(context.exception))
    
    def test_connection_check(self):
        """Test the connection check method."""
        # Mock the server_info method
        self.mock_client.server_info.return_value = {"version": "4.4.0"}
        
        # Call the method
        result = self.db_service.check_connection()
        
        # Check the result
        self.assertTrue(result['connected'])
        self.assertEqual(result['version'], "4.4.0")
    
    def test_connection_check_failure(self):
        """Test the connection check method when connection fails."""
        # Make server_info raise an exception
        self.mock_client.server_info.side_effect = ServerSelectionTimeoutError("Could not connect")
        
        # Call the method
        result = self.db_service.check_connection()
        
        # Check the result
        self.assertFalse(result['connected'])
        self.assertIn("Could not connect", result['error'])


if __name__ == "__main__":
    import unittest
    unittest.main()
