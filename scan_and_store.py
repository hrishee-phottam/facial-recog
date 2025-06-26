"""
scan_and_store.py - Face recognition image scanner and MongoDB storage system

This module provides functionality for scanning images using an API service and storing
the results in MongoDB. It includes robust error handling, retry mechanisms, and detailed
logging with emojis for better visualization.

Key Features:
- 📸 Image scanning with retry mechanism
- 🔐 Secure MongoDB integration
- 📊 Progress tracking and statistics
- 🔄 Automatic retries for failed operations
- 📝 Detailed logging with emojis

Environment Variables:
- IMAGES_DIR: Path to directory containing images to process
- API_URL: URL of the face recognition API service
- API_MAX_RETRIES: Maximum number of retries for API calls
- API_RETRY_DELAY: Delay between retry attempts
- MONGODB_DB_NAME: Name of the MongoDB database
- MONGODB_COLLECTION_NAME: Name of the MongoDB collection
- MONGODB_USERNAME: MongoDB username
- MONGODB_PASSWORD: MongoDB password
"""

import argparse
import os
import logging
import requests
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from urllib.parse import quote_plus
from datetime import datetime
import time
import json
from typing import Any, Dict, Optional
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

def scan_image(file_path: str, url: str, max_retries: int = 3, retry_delay: float = 2) -> Optional[Dict[str, Any]]:
    """
    📸 Scan an image file using the face recognition API service

    This function sends an image file to the specified API endpoint and returns the parsed
    JSON response. It implements a robust retry mechanism to handle network errors and
    temporary failures.

    Args:
        file_path (str): Path to the image file to be scanned
        url (str): URL of the face recognition API endpoint
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        retry_delay (float, optional): Delay between retry attempts in seconds. Defaults to 2.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON response from the API, or None if all retries fail

    Raises:
        requests.exceptions.RequestException: If there's a network error
        json.JSONDecodeError: If the API response cannot be parsed as JSON
    """
    for attempt in range(max_retries):
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(url, files={'file': f}, timeout=None)
                response.raise_for_status()
                return response.json()
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
            logging.error(f"Response status code: {response.status_code}")
            logging.error(f"Response body: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error for {file_path}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response for {file_path}: {str(e)}")
            logging.error(f"Raw response: {response.text}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error while processing {file_path}: {str(e)}")
            return None
        finally:
            try:
                if 'response' in locals():
                    response.close()
            except:
                pass


def main() -> None:
    """
    🚀 Main entry point for the image scanning and storage system

    This function orchestrates the entire image scanning and storage process:
    1. 📄 Loads configuration from environment variables
    2. 🔐 Establishes MongoDB connection
    3. 📊 Counts and processes images
    4. 📝 Logs detailed progress and statistics

    Environment Variables Required:
        - IMAGES_DIR: Path to directory containing images
        - API_URL: URL of the face recognition API
        - API_MAX_RETRIES: Maximum retry attempts for API calls
        - API_RETRY_DELAY: Delay between retry attempts
        - MONGODB_DB_NAME: MongoDB database name
        - MONGODB_COLLECTION_NAME: MongoDB collection name
        - MONGODB_USERNAME: MongoDB username
        - MONGODB_PASSWORD: MongoDB password

    Raises:
        ValueError: If required environment variables are missing
        Exception: If MongoDB connection fails
    """
    # Get configuration from environment variables (no fallback values for required settings)
    required_vars = [
        'IMAGES_DIR',
        'API_URL',
        'API_MAX_RETRIES',
        'API_RETRY_DELAY',
        'MONGODB_DB_NAME',
        'MONGODB_COLLECTION_NAME',
        'MONGODB_USERNAME',
        'MONGODB_PASSWORD'
    ]
    
    # Check for missing required environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Load configuration from environment variables
    path = os.environ['IMAGES_DIR']
    url = os.environ['API_URL']
    max_retries = int(os.environ['API_MAX_RETRIES'])
    retry_delay = float(os.environ['API_RETRY_DELAY'])
    db_name = os.environ['MONGODB_DB_NAME']
    collection_name = os.environ['MONGODB_COLLECTION_NAME']
    username = os.environ['MONGODB_USERNAME']
    password = os.environ['MONGODB_PASSWORD']
    
    # MongoDB Atlas credentials from environment
    username = os.getenv('MONGODB_USERNAME')
    password = os.getenv('MONGODB_PASSWORD')
    
    # Validate required environment variables
    required_vars = ['API_URL', 'MONGODB_DB_NAME', 'MONGODB_USERNAME', 'MONGODB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate MongoDB credentials
    if not username or not password:
        raise ValueError("MongoDB credentials (MONGODB_USERNAME and MONGODB_PASSWORD) must not be empty")
    
    # Log the MongoDB connection attempt
    logging.info(f"Attempting to connect to MongoDB with database: {db_name}")
    
    # Use the exact Atlas connection string format with properly escaped credentials
    mongo_uri = os.getenv('MONGODB_URI', f'mongodb+srv://{quote_plus(username)}:{quote_plus(password)}@cluster0.s35kdmn.mongodb.net/{db_name}?retryWrites=true&w=majority&appName=Cluster0')
    
    # Create client with Server API version 1
    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        # Test the connection
        client.admin.command('ping')
        logging.info("Successfully connected to MongoDB")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        logging.error(f"MongoDB URI used: {mongo_uri}")
        logging.error("Please check your MongoDB credentials in .env file")
        raise
    
    # Verify connection
    try:
        client.admin.command('ping')
        logging.info("Successfully connected to MongoDB Atlas!")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise
    
    # Get database and collection
    db = client[db_name]
    collection = db[collection_name]

    # Configure enhanced logging with emojis and colors
    class ColoredFormatter(logging.Formatter):
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

    # Configure logging with enhanced format
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('scan_and_store.log')
        ]
    )
    
    # Set up colored formatter for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logging.getLogger().handlers[0] = console_handler

    # Connect to MongoDB with detailed logging
    try:
        client = MongoClient(mongo_uri)
        logging.info(f"Connected to MongoDB, server info: {client.server_info()}")
        db = client[db_name]
        collection = db[collection_name]
        logging.info(f"Using database: {db_name}, collection: {collection_name}")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

    # Count total images to process
    total_images = sum(1 for root, _, files in os.walk(path)
                      for name in files if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')))
    processed = 0
    successful = 0
    failed = 0

    logging.info('🔍 Found %d images to process in directory: %s', total_images, path)

    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith((
                '.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                file_path = os.path.join(root, name)
                try:
                    logging.info('📸 Processing image %d/%d: %s', processed + 1, total_images, file_path)
                    
                    # Get file size for logging
                    file_size = os.path.getsize(file_path)
                    logging.info('📁 File size: %.2f MB', file_size/1024/1024)
                    
                    # Scan the image
                    data = scan_image(file_path, url)
                    if data is None:
                        failed += 1
                        logging.error(f'Failed to process {file_path}')
                        continue
                    
                    data['filename'] = name
                    data['processed_at'] = datetime.now().isoformat()
                    
                    try:
                        # Log the data being inserted
                        logging.debug(f"Attempting to insert data: {json.dumps(data, indent=2)}")
                        
                        # Insert the data
                        result = collection.insert_one(data)
                        successful += 1
                        logging.info(f'Successfully processed and stored {file_path}, MongoDB ID: {str(result.inserted_id)}')
                        
                        # Verify the insert by finding the document
                        inserted_doc = collection.find_one({'_id': result.inserted_id})
                        if inserted_doc:
                            logging.info(f"Verified document exists in MongoDB: {json.dumps(inserted_doc, indent=2)}")
                        else:
                            logging.warning(f"Document with ID {str(result.inserted_id)} not found after insertion!")
                    except Exception as e:
                        failed += 1
                        logging.error(f'Failed to store data for {file_path} in MongoDB: {str(e)}')
                        # Convert ObjectId to string for JSON serialization
                        data_to_log = {k: str(v) if isinstance(v, ObjectId) else v for k, v in data.items()}
                        logging.error(f'Data that failed to store: {json.dumps(data_to_log, indent=2)}')
                except Exception as exc:
                    failed += 1
                    logging.error(f'Failed to process {file_path}: {str(exc)}')
                    logging.error(f'Exception type: {type(exc).__name__}')
                    logging.error(f'System error info: {sys.exc_info()}')
                finally:
                    processed += 1
                    logging.info('Progress: %d/%d processed, %d successful, %d failed', 
                                processed, total_images, successful, failed)
                    
    # Log final summary
    logging.info('\n=== Processing Summary ===')
    logging.info(f'Total images processed: {processed}')
    logging.info(f'Successful scans: {successful}')
    logging.info(f'Failed scans: {failed}')
    logging.info(f'Success rate: {successful/processed*100:.2f}%')
    logging.info('=== End of Processing ===')


if __name__ == '__main__':
    main()
