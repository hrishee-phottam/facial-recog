"""
Tests for the settings module.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from src.config.settings import Settings, get_settings


class TestSettings(TestCase):
    """Test cases for the Settings class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_env = {
            'API_URL': 'https://api.example.com',
            'MONGODB_URI': 'mongodb://localhost:27017',
            'MONGODB_DB_NAME': 'test_db',
            'MONGODB_COLLECTION_NAME': 'test_collection',
            'IMAGES_DIR': '/test/images',
            'SUPPORTED_EXTENSIONS': '.jpg,.jpeg,.png',
            'LOG_LEVEL': 'DEBUG',
            'LOG_FILE': 'test.log',
        }
        
        # Patch environment variables
        self.env_patcher = mock.patch.dict(os.environ, self.test_env)
        self.env_patcher.start()
        
        # Clear the settings cache
        from src.config import settings as settings_module
        settings_module._settings = None
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    def test_settings_initialization(self):
        """Test that settings are loaded correctly from environment variables."""
        settings = get_settings()
        
        self.assertEqual(str(settings.API_URL), self.test_env['API_URL'])
        self.assertEqual(settings.MONGODB_URI, self.test_env['MONGODB_URI'])
        self.assertEqual(settings.MONGODB_DB_NAME, self.test_env['MONGODB_DB_NAME'])
        self.assertEqual(settings.MONGODB_COLLECTION_NAME, self.test_env['MONGODB_COLLECTION_NAME'])
        self.assertEqual(settings.IMAGES_DIR, self.test_env['IMAGES_DIR'])
        self.assertEqual(settings.SUPPORTED_EXTENSIONS, ['.jpg', '.jpeg', '.png'])
        self.assertEqual(settings.LOG_LEVEL, 'DEBUG')
        self.assertEqual(settings.LOG_FILE, 'test.log')
    
    def test_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        self.assertIs(settings1, settings2)
    
    def test_mongodb_connection_params(self):
        """Test MongoDB connection parameters."""
        settings = get_settings()
        conn_params = settings.mongodb_connection_params
        
        self.assertEqual(conn_params['host'], settings.MONGODB_URI)
        self.assertEqual(conn_params['connectTimeoutMS'], settings.MONGODB_CONNECT_TIMEOUT_MS)
        self.assertEqual(conn_params['serverSelectionTimeoutMS'], settings.MONGODB_SERVER_SELECTION_TIMEOUT_MS)
        self.assertTrue(conn_params['retryWrites'])
        self.assertEqual(conn_params['w'], 'majority')
    
    def test_supported_extensions_str(self):
        """Test supported extensions string representation."""
        settings = get_settings()
        self.assertEqual(settings.supported_extensions_str, ".jpg, .jpeg, .png")
    
    def test_validate_mongodb_uri(self):
        """Test MongoDB URI validation."""
        # Valid URIs
        valid_uris = [
            'mongodb://localhost:27017',
            'mongodb+srv://user:pass@cluster.example.com/test',
        ]
        
        for uri in valid_uris:
            with self.subTest(uri=uri):
                settings = Settings(MONGODB_URI=uri, **self.test_env)
                self.assertEqual(settings.MONGODB_URI, uri)
        
        # Invalid URIs
        invalid_uris = [
            'invalid_uri',
            'http://example.com',
            'ftp://example.com',
        ]
        
        for uri in invalid_uris:
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                Settings(MONGODB_URI=uri, **self.test_env)
    
    def test_validate_extensions(self):
        """Test file extension validation."""
        test_cases = [
            (['.jpg', '.jpeg', '.png'], ['.jpg', '.jpeg', '.png']),  # Already correct
            (['JPG', 'JPEG', 'PNG'], ['.jpg', '.jpeg', '.png']),      # Uppercase
            (['.JPG', '.JPEG', '.PNG'], ['.jpg', '.jpeg', '.png']),   # Uppercase with dot
            (['jpg', 'jpeg', 'png'], ['.jpg', '.jpeg', '.png']),      # No dot
            (None, ['.jpg', '.jpeg', '.png']),                        # None falls back to default
        ]
        
        for input_exts, expected in test_cases:
            with self.subTest(input_exts=input_exts):
                settings = Settings(SUPPORTED_EXTENSIONS=input_exts, **self.test_env)
                self.assertEqual(settings.SUPPORTED_EXTENSIONS, expected)
    
    def test_validate_log_level(self):
        """Test log level validation."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            with self.subTest(level=level):
                settings = Settings(LOG_LEVEL=level.lower(), **self.test_env)
                self.assertEqual(settings.LOG_LEVEL, level)
        
        # Invalid log level
        with self.assertRaises(ValueError):
            Settings(LOG_LEVEL='INVALID', **self.test_env)


if __name__ == "__main__":
    import unittest
    unittest.main()
