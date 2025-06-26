"""
Tests for the ImageProcessor class.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import MagicMock, patch, call

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from src.core.processor import ImageProcessor, ProcessingResult
from src.config.settings import Settings


class TestImageProcessor(TestCase):
    """Test cases for the ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test settings object
        self.settings = Settings(
            API_URL="https://api.example.com",
            MONGODB_URI="mongodb://localhost:27017",
            MONGODB_DB_NAME="test_db",
            MONGODB_COLLECTION_NAME="test_collection",
            IMAGES_DIR=tempfile.mkdtemp(),
            SUPPORTED_EXTENSIONS=['.jpg', '.jpeg', '.png'],
            LOG_LEVEL="WARNING",
            LOG_FILE="test_processor.log"
        )
        
        # Create a test image file
        self.test_image_path = os.path.join(self.settings.IMAGES_DIR, "test.jpg")
        with open(self.test_image_path, 'wb') as f:
            f.write(b'fake image data')
        
        # Initialize the processor with mock services
        self.mock_api_service = MagicMock()
        self.mock_db_service = MagicMock()
        
        self.processor = ImageProcessor(
            settings=self.settings,
            api_service=self.mock_api_service,
            db_service=self.mock_db_service
        )
    
    def test_initialization(self):
        """Test that the processor initializes correctly."""
        self.assertEqual(self.processor.settings, self.settings)
        self.assertEqual(self.processor.api_service, self.mock_api_service)
        self.assertEqual(self.processor.db_service, self.mock_db_service)
        self.assertEqual(len(self.processor._observers), 0)
    
    def test_register_observer(self):
        """Test registering an observer."""
        observer = MagicMock()
        self.processor.register_observer(observer)
        
        self.assertIn(observer, self.processor._observers)
        self.assertEqual(len(self.processor._observers), 1)
    
    def test_notify_observers(self):
        """Test notifying observers of an event."""
        # Register mock observers
        observer1 = MagicMock()
        observer2 = MagicMock()
        self.processor.register_observer(observer1)
        self.processor.register_observer(observer2)
        
        # Notify observers
        test_data = {"test": "data"}
        self.processor._notify_observers("test_event", test_data)
        
        # Check that both observers were called
        observer1.assert_called_once_with("test_event", test_data)
        observer2.assert_called_once_with("test_event", test_data)
    
    @patch('src.core.processor.os.path.getsize')
    @patch('src.core.processor.os.path.isfile')
    def test_validate_image_file(self, mock_isfile, mock_getsize):
        """Test image file validation."""
        # Test valid file
        mock_isfile.return_value = True
        mock_getsize.return_value = 5 * 1024 * 1024  # 5MB
        
        result = self.processor._validate_image_file("test.jpg")
        self.assertTrue(result)
        
        # Test non-existent file
        mock_isfile.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.processor._validate_image_file("nonexistent.jpg")
        
        # Test unsupported extension
        mock_isfile.return_value = True
        with self.assertRaises(ValueError):
            self.processor._validate_image_file("test.txt")
        
        # Test file too large
        mock_isfile.return_value = True
        mock_getsize.return_value = 15 * 1024 * 1024  # 15MB > 10MB limit
        with self.assertRaises(ValueError):
            self.processor._validate_image_file("test.jpg")
    
    @patch('src.core.processor.ImageProcessor._validate_image_file')
    def test_process_image_success(self, mock_validate):
        """Test successful image processing."""
        # Set up mocks
        mock_validate.return_value = True
        
        test_result = {"result": [{"face_id": 1, "confidence": 0.95}]}
        self.mock_api_service.scan_image.return_value = test_result
        self.mock_db_service.save_result.return_value = "test_id"
        
        # Register a mock observer
        mock_observer = MagicMock()
        self.processor.register_observer(mock_observer)
        
        # Process the image
        result = self.processor.process_image(self.test_image_path)
        
        # Check the result
        self.assertIsInstance(result, ProcessingResult)
        self.assertTrue(result.success)
        self.assertEqual(result.file_path, self.test_image_path)
        self.assertEqual(result.result, test_result)
        
        # Check that the API was called with the correct arguments
        self.mock_api_service.scan_image.assert_called_once()
        
        # Check that the result was saved to the database
        self.mock_db_service.save_result.assert_called_once_with(
            file_path=self.test_image_path,
            result=test_result,
            success=True,
            error_message=None
        )
        
        # Check that the observer was notified
        mock_observer.assert_called_once()
    
    @patch('src.core.processor.ImageProcessor._validate_image_file')
    def test_process_image_api_error(self, mock_validate):
        """Test image processing with API error."""
        # Set up mocks
        mock_validate.return_value = True
        
        # Simulate API error
        self.mock_api_service.scan_image.side_effect = Exception("API error")
        
        # Process the image
        result = self.processor.process_image(self.test_image_path)
        
        # Check the result
        self.assertIsInstance(result, ProcessingResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.file_path, self.test_image_path)
        
        # Check that the API was called
        self.mock_api_service.scan_image.assert_called_once()
        
        # Check that the error was saved to the database
        self.mock_db_service.save_result.assert_called_once()
        call_args = self.mock_db_service.save_result.call_args[1]
        self.assertEqual(call_args['file_path'], self.test_image_path)
        self.assertFalse(call_args['success'])
        self.assertIn("API error", call_args['error_message'])
    
    @patch('src.core.processor.ImageProcessor._validate_image_file')
    @patch('src.core.processor.glob.glob')
    def test_process_directory(self, mock_glob, mock_validate):
        """Test processing a directory of images."""
        # Set up mocks
        mock_validate.return_value = True
        
        # Create test files
        test_files = [
            os.path.join(self.settings.IMAGES_DIR, f"test_{i}.jpg") for i in range(3)
        ]
        mock_glob.return_value = test_files
        
        # Set up API to return different results for each file
        self.mock_api_service.scan_image.side_effect = [
            {"result": [{"face_id": i, "confidence": 0.9}]} for i in range(3)
        ]
        
        # Process the directory
        results = self.processor.process_directory(self.settings.IMAGES_DIR)
        
        # Check the results
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, ProcessingResult) for r in results))
        self.assertTrue(all(r.success for r in results))
        
        # Check that each file was processed
        self.assertEqual(self.mock_api_service.scan_image.call_count, 3)
        for i, file_path in enumerate(test_files):
            self.mock_api_service.scan_image.assert_any_call(file_path)
        
        # Check that results were saved to the database
        self.assertEqual(self.mock_db_service.save_result.call_count, 3)
    
    def test_process_directory_nonexistent(self):
        """Test processing a non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_directory("/nonexistent/directory")
    
    @patch('src.core.processor.os.path.isdir')
    @patch('src.core.processor.glob.glob')
    def test_process_directory_no_images(self, mock_glob, mock_isdir):
        """Test processing a directory with no images."""
        mock_isdir.return_value = True
        mock_glob.return_value = []
        
        with self.assertRaises(ValueError) as context:
            self.processor.process_directory(self.settings.IMAGES_DIR)
        
        self.assertIn("No image files found", str(context.exception))


if __name__ == "__main__":
    import unittest
    unittest.main()
