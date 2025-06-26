import os
import unittest
from unittest.mock import patch, MagicMock
from scan_and_store import scan_image, main
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

class TestScanAndStore(unittest.TestCase):
    @patch('requests.post')
    def test_scan_image_success(self, mock_post):
        """
        Test successful image scanning with valid response
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_post.return_value = mock_response
        
        result = scan_image('test.jpg', 'http://test.com/api')
        self.assertIsNotNone(result)
        self.assertEqual(result, {'success': True})

    @patch('requests.post')
    def test_scan_image_connection_error(self, mock_post):
        """
        Test image scanning with connection error and retry mechanism
        """
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        result = scan_image('test.jpg', 'http://test.com/api')
        self.assertIsNone(result)

    @patch('requests.post')
    def test_scan_image_timeout(self, mock_post):
        """
        Test image scanning with timeout error
        """
        mock_post.side_effect = requests.exceptions.Timeout()
        
        result = scan_image('test.jpg', 'http://test.com/api')
        self.assertIsNone(result)

    @patch('requests.post')
    def test_scan_image_json_decode_error(self, mock_post):
        """
        Test image scanning with invalid JSON response
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError('', '', 0)
        mock_response.text = 'invalid json'
        mock_post.return_value = mock_response
        
        result = scan_image('test.jpg', 'http://test.com/api')
        self.assertIsNone(result)

    @patch.dict(os.environ, {
        'IMAGES_DIR': '/test/images',
        'API_URL': 'http://test.com/api',
        'API_MAX_RETRIES': '3',
        'API_RETRY_DELAY': '2',
        'MONGODB_DB_NAME': 'test_db',
        'MONGODB_COLLECTION_NAME': 'test_collection',
        'MONGODB_USERNAME': 'test_user',
        'MONGODB_PASSWORD': 'test_pass'
    })
    @patch('os.walk')
    @patch('os.path.getsize')
    @patch('datetime.datetime')
    def test_main(self, mock_datetime, mock_getsize, mock_walk):
        """
        Test main function with mocked file system and MongoDB
        """
        # Mock file system
        mock_walk.return_value = [('/test/images', [], ['test.jpg'])]
        mock_getsize.return_value = 1024  # 1KB
        
        # Mock datetime
        mock_datetime.now.return_value = datetime(2024, 1, 1)
        
        # Mock MongoDB operations
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        with patch('pymongo.MongoClient', return_value=mock_client):
            # Mock scan_image to return a successful response
            with patch('scan_and_store.scan_image', return_value={'success': True}):
                main()
                
                # Verify MongoDB operations
                mock_collection.insert_one.assert_called_once()
                
                # Verify logging was called
                mock_client.admin.command.assert_called_with('ping')

if __name__ == '__main__':
    unittest.main()
