"""
Tests for the APIService class.
"""
import os
import sys
import json
from unittest import TestCase, mock
from unittest.mock import MagicMock, patch, call

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from src.services.api_service import APIService, APIError
from src.config.settings import Settings


class TestAPIService(TestCase):
    """Test cases for the APIService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test settings object
        self.settings = Settings(
            API_URL="https://api.example.com",
            API_MAX_RETRIES=3,
            API_RETRY_DELAY=0.1,  # Shorter delay for tests
            API_TIMEOUT=5,
            LOG_LEVEL="WARNING"
        )
        
        # Initialize the API service
        self.api_service = APIService(settings=self.settings)
    
    @patch('src.services.api_service.requests.Session')
    def test_initialization(self, mock_session):
        """Test that the API service initializes correctly."""
        self.assertEqual(self.api_service.base_url, str(self.settings.API_URL))
        self.assertEqual(self.api_service.max_retries, self.settings.API_MAX_RETRIES)
        self.assertEqual(self.api_service.retry_delay, self.settings.API_RETRY_DELAY)
        self.assertEqual(self.api_service.timeout, self.settings.API_TIMEOUT)
        
        # Test default settings
        default_service = APIService()
        self.assertEqual(default_service.max_retries, 3)
        self.assertEqual(default_service.retry_delay, 1.0)
    
    @patch('src.services.api_service.requests.Session')
    def test_scan_image_success(self, mock_session):
        """Test successful image scanning."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": [{"face_id": 1, "confidence": 0.95}]}
        
        # Set up session mock
        mock_session.return_value.__enter__.return_value.post.return_value = mock_response
        
        # Call the method
        test_file = "test.jpg"
        with open(test_file, 'wb') as f:
            f.write(b'test image data')
        
        try:
            result = self.api_service.scan_image(test_file)
            
            # Check the result
            self.assertEqual(result, {"result": [{"face_id": 1, "confidence": 0.95}]})
            
            # Check that the request was made correctly
            mock_session.return_value.__enter__.return_value.post.assert_called_once()
            args, kwargs = mock_session.return_value.__enter__.return_value.post.call_args
            
            self.assertIn('files', kwargs)
            self.assertEqual(kwargs['files']['file'][0], 'test.jpg')
            self.assertEqual(kwargs['timeout'], self.settings.API_TIMEOUT)
            
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.services.api_service.requests.Session')
    @patch('src.services.api_service.time.sleep')
    def test_scan_image_retry_on_failure(self, mock_sleep, mock_session):
        """Test that the service retries on temporary failures."""
        # Set up mock to fail twice then succeed
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"result": []}
        
        mock_response_failure = MagicMock()
        mock_response_failure.status_code = 500
        
        mock_session.return_value.__enter__.return_value.post.side_effect = [
            RequestException("Connection error"),
            mock_response_failure,
            mock_response_success
        ]
        
        # Call the method
        test_file = "test_retry.jpg"
        with open(test_file, 'wb') as f:
            f.write(b'test image data')
        
        try:
            result = self.api_service.scan_image(test_file)
            
            # Should eventually succeed
            self.assertEqual(result, {"result": []})
            
            # Should have been called 3 times (2 failures + 1 success)
            self.assertEqual(mock_session.return_value.__enter__.return_value.post.call_count, 3)
            
            # Should have slept between retries
            self.assertEqual(mock_sleep.call_count, 2)
            mock_sleep.assert_called_with(self.settings.API_RETRY_DELAY)
            
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.services.api_service.requests.Session')
    @patch('src.services.api_service.time.sleep')
    def test_scan_image_max_retries_exceeded(self, mock_sleep, mock_session):
        """Test that the service gives up after max retries."""
        # Set up mock to always fail
        mock_session.return_value.__enter__.return_value.post.side_effect = RequestException("Connection error")
        
        # Call the method
        test_file = "test_failure.jpg"
        with open(test_file, 'wb') as f:
            f.write(b'test image data')
        
        try:
            with self.assertRaises(APIError) as context:
                self.api_service.scan_image(test_file)
            
            # Check the error message
            self.assertIn("Max retries (3) exceeded", str(context.exception))
            
            # Should have been called max_retries + 1 times
            self.assertEqual(
                mock_session.return_value.__enter__.return_value.post.call_count,
                self.settings.API_MAX_RETRIES + 1
            )
            
            # Should have slept between retries
            self.assertEqual(mock_sleep.call_count, self.settings.API_MAX_RETRIES)
            
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.services.api_service.requests.Session')
    def test_scan_image_http_error(self, mock_session):
        """Test handling of HTTP errors."""
        # Set up mock to return an error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid image"}
        mock_response.raise_for_status.side_effect = requests.HTTPError("Bad Request")
        
        mock_session.return_value.__enter__.return_value.post.return_value = mock_response
        
        # Call the method
        test_file = "test_error.jpg"
        with open(test_file, 'wb') as f:
            f.write(b'test image data')
        
        try:
            with self.assertRaises(APIError) as context:
                self.api_service.scan_image(test_file)
            
            # Check the error message
            self.assertIn("API request failed with status 400", str(context.exception))
            
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.services.api_service.requests.Session')
    def test_scan_image_invalid_json(self, mock_session):
        """Test handling of invalid JSON response."""
        # Set up mock to return invalid JSON
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        mock_session.return_value.__enter__.return_value.post.return_value = mock_response
        
        # Call the method
        test_file = "test_invalid.jpg"
        with open(test_file, 'wb') as f:
            f.write(b'test image data')
        
        try:
            with self.assertRaises(APIError) as context:
                self.api_service.scan_image(test_file)
            
            # Check the error message
            self.assertIn("Failed to decode JSON response", str(context.exception))
            
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == "__main__":
    import unittest
    unittest.main()
