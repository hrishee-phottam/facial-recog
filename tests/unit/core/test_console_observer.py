"""
Tests for the ConsoleObserver class.
"""
import os
import sys
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from src.core.console_observer import ConsoleObserver


class TestConsoleObserver(TestCase):
    """Test cases for the ConsoleObserver class."""

    def setUp(self):
        """Set up test fixtures."""
        self.console_observer = ConsoleObserver(
            show_progress=True,
            show_summary=True
        )
        
        # Mock the console to capture output
        self.console_observer.console = MagicMock()
    
    def test_initialization(self):
        """Test that the observer initializes correctly."""
        self.assertEqual(self.console_observer.processed, 0)
        self.assertEqual(self.console_observer.successful, 0)
        self.assertEqual(self.console_observer.failed, 0)
        self.assertTrue(self.console_observer.show_progress)
        self.assertTrue(self.console_observer.show_summary)
    
    def test_call_success(self):
        """Test handling a successful processing event."""
        file_path = "/path/to/image.jpg"
        result = {
            "success": True,
            "result": [
                {"face_id": 1, "confidence": 0.95},
                {"face_id": 2, "confidence": 0.87}
            ]
        }
        
        self.console_observer(file_path, result)
        
        # Check that the success was recorded
        self.assertEqual(self.console_observer.processed, 1)
        self.assertEqual(self.console_observer.successful, 1)
        self.assertEqual(self.console_observer.failed, 0)
        
        # Check that the console was called with the expected output
        self.console_observer.console.print.assert_called()
    
    def test_call_error(self):
        """Test handling a failed processing event."""
        file_path = "/path/to/image.jpg"
        result = {"success": False, "error": "Face not found"}
        error = Exception("Face not found")
        
        self.console_observer(file_path, result, error)
        
        # Check that the failure was recorded
        self.assertEqual(self.console_observer.processed, 1)
        self.assertEqual(self.console_observer.successful, 0)
        self.assertEqual(self.console_observer.failed, 1)
        
        # Check that the console was called with the error message
        self.console_observer.console.print.assert_called_with(
            f"✗ {os.path.basename(file_path)} - Error: {str(error)}",
            style="red"
        )
    
    def test_display_summary(self):
        """Test the display_summary method."""
        # Process some test data
        self.console_observer.processed = 10
        self.console_observer.successful = 8
        self.console_observer.failed = 2
        
        # Call the method
        self.console_observer.display_summary()
        
        # Check that the console was called with a table
        self.console_observer.console.print.assert_called()
        
        # Check that the summary panel was created
        self.console_observer.console.print.assert_called()
    
    def test_get_progress_tracker(self):
        """Test the get_progress_tracker method."""
        # Call the method
        progress = self.console_observer.get_progress_tracker(100)
        
        # Check that a Progress instance was returned
        self.assertIsNotNone(progress)
        self.assertEqual(progress.tasks[0].total, 100)


if __name__ == "__main__":
    import unittest
    unittest.main()
