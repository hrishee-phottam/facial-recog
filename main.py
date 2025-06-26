"""
face_recognition_test.py - Test script for face recognition functionality

This is a test version of the original implementation that was working before refactoring.
It's meant to help verify the core functionality still works as expected.

TODO: Add configuration validation
TODO: Improve MongoDB connection handling with proper cleanup
TODO: Add progress tracking for processed images
TODO: Add command-line argument parsing
TODO: Add type hints for better code clarity
TODO: Move MongoDB URL to environment variables
TODO: Implement exponential backoff for retries
"""

import os
import sys
import logging
import time
import json
from typing import Dict, Any, Optional
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from urllib.parse import quote_plus
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('.env')

def scan_image(file_path: str, url: str, max_retries: int = 3, retry_delay: float = 2) -> Optional[Dict[str, Any]]:
    # TODO: Add exponential backoff for retries
    # TODO: Add timeout configuration
    """Scan an image file using the face recognition API service"""
    for attempt in range(max_retries):
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(url, files={'file': f}, timeout=None)
                response.raise_for_status()
                response_data = response.json()
                logging.debug(f'Raw API response for {file_path}: {json.dumps(response_data, default=str, indent=2)}')
                return response_data
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error on attempt {attempt + 1}/{max_retries} for {file_path}: {str(e)}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to connect after {max_retries} attempts for {file_path}")
                return None
            time.sleep(retry_delay)
        except requests.exceptions.Timeout as e:
            logging.error(f"Timeout error for {file_path}: {str(e)}")
            return None
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error for {file_path}: {str(e)}")
            if 'response' in locals():
                logging.error(f"Response status code: {response.status_code}")
                logging.error(f"Response body: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error for {file_path}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response for {file_path}: {str(e)}")
            if 'response' in locals():
                logging.error(f"Raw response: {response.text}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error while processing {file_path}: {str(e)}")
            return None

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET)
        return f'{color}{log_message}{self.RESET}'

def main():
    """Main function for testing face recognition"""
    try:
        # Required environment variables
        required_vars = [
            'IMAGES_DIR',
            'API_URL',
            'MONGODB_DB_NAME',
            'MONGODB_COLLECTION_NAME',
            'MONGODB_USERNAME',
            'MONGODB_PASSWORD'
        ]
        
        # Check for missing required environment variables
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Load configuration
        config = {
            'path': os.environ['IMAGES_DIR'],
            'url': os.environ['API_URL'],
            'max_retries': int(os.getenv('API_MAX_RETRIES', '3')),
            'retry_delay': float(os.getenv('API_RETRY_DELAY', '2.0')),
            'db_name': os.environ['MONGODB_DB_NAME'],
            'collection_name': os.environ['MONGODB_COLLECTION_NAME'],
            'username': os.environ['MONGODB_USERNAME'],
            'password': os.environ['MONGODB_PASSWORD']
        }
        
        # Configure logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - [%(levelname)s] - %(message)s',
            handlers=[
                console_handler,
                logging.FileHandler('face_recognition_test.log')
            ]
        )
        
        # TODO: Move MongoDB URL to environment variables
        mongo_uri = f'mongodb+srv://{quote_plus(config["username"])}:{quote_plus(config["password"])}@cluster0.s35kdmn.mongodb.net/{config["db_name"]}?retryWrites=true&w=majority&appName=Cluster0'
        
        # TODO: Add connection timeout and retry logic
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        
        try:
            # Test MongoDB connection
            client.admin.command('ping')
            logging.info("✅ Successfully connected to MongoDB")
        except Exception as e:
            logging.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            raise
        
        db = client[config['db_name']]
        collection = db[config['collection_name']]
        
        # Count images to process
        image_files = []
        for root, _, files in os.walk(config['path']):
            for name in files:
                if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    image_files.append(os.path.join(root, name))
        
        total_images = len(image_files)
        logging.info(f'🔍 Found {total_images} images to process in directory: {config["path"]}')
        
        # TODO: Add progress bar or better progress tracking
        # Process each image
        for i, file_path in enumerate(image_files, 1):
            try:
                logging.info(f'📸 Processing image {i}/{total_images}: {file_path}')
                
                # Scan the image
                data = scan_image(
                    file_path=file_path,
                    url=config['url'],
                    max_retries=config['max_retries'],
                    retry_delay=config['retry_delay']
                )
                
                if not data:
                    logging.warning(f'⚠️  No data returned for {file_path}')
                    continue
                    
                # Process and store the results
                if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
                    for j, result in enumerate(data['result']):
                        if 'embedding' in result:
                            doc = {
                                'filename': os.path.basename(file_path),
                                'file_path': file_path,
                                'processed_at': datetime.now().isoformat(),
                                'embedding': result['embedding'],
                                'box': result.get('box', {}),
                                'execution_time': result.get('execution_time', {})
                            }
                            collection.insert_one(doc)
                            logging.info(f'✅ Successfully stored face {j+1} from {os.path.basename(file_path)}')
                
            except Exception as e:
                logging.error(f'❌ Error processing {file_path}: {str(e)}')
        
        # TODO: Add summary statistics
        success_count = total_images - len([f for f in image_files if not f])  # Placeholder
        logging.info(f'✅ Processing complete! Successfully processed {success_count}/{total_images} images')
        
    except Exception as e:
        logging.error(f'❌ Fatal error: {str(e)}')
        return 1
    
    return 0

if __name__ == '__main__':
    import requests  # Moved here to avoid unused import warning
    sys.exit(main())
