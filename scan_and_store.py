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
                response_data = response.json()
                # Log the raw response for debugging
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
                        
                    # Save the first API response to a file for inspection
                    if not os.path.exists('api_response.json'):
                        with open('api_response.json', 'w') as f:
                            json.dump(data, f, indent=2, default=str)
                        logging.info(f'Saved API response to api_response.json')
                    
                    # Log the type and keys of the response
                    logging.info(f'Response type: {type(data).__name__}')
                    if isinstance(data, dict):
                        logging.info(f'Top-level keys: {list(data.keys())}')
                        if 'result' in data:
                            logging.info(f'Result type: {type(data["result"]).__name__}')
                            if isinstance(data['result'], list):
                                logging.info(f'Number of results: {len(data["result"])}')
                                for i, result in enumerate(data['result'][:3]):  # Log first 3 results max
                                    if isinstance(result, dict):
                                        logging.info(f'Result {i} keys: {list(result.keys())}')
                                        if 'face_landmarks' in result:
                                            logging.info(f'face_landmarks type: {type(result["face_landmarks"]).__name__}')
                                            if isinstance(result['face_landmarks'], list):
                                                logging.info(f'face_landmarks length: {len(result["face_landmarks"])}')
                                    else:
                                        logging.info(f'Result {i} is not a dictionary')
                    
                    # Initialize face_embeddings list
                    face_embeddings = []
                    
                    # Extract face embeddings from the response
                    if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
                        for result in data['result']:
                            if isinstance(result, dict) and 'embedding' in result and result['embedding']:
                                # Create a document for each face embedding
                                face_embeddings.append({
                                    'embedding': result['embedding'],
                                    'box': result.get('box', {}),
                                    'execution_time': result.get('execution_time', {})
                                })
                    
                    # Log the number of faces found
                    logging.info(f'Found {len(face_embeddings)} face(s) in {file_path}')
                    
                    if not face_embeddings:
                        logging.warning(f'No face embeddings found in {file_path}')
                        continue
                    
                    # Add common metadata
                    base_data = {
                        'filename': name,
                        'file_path': file_path,
                        'processed_at': datetime.now().isoformat(),
                        'face_count': len(face_embeddings),
                        'calculator_version': data.get('calculator_version', 'unknown')
                    }
                    
                    # Create a document for each face embedding
                    embedding_docs = []
                    for i, face_data in enumerate(face_embeddings):
                        embedding_doc = {
                            **base_data,
                            'embedding': face_data['embedding'],
                            'embedding_index': i,
                            'embedding_length': len(face_data['embedding']) if face_data['embedding'] else 0,
                            'execution_time': face_data.get('execution_time', {})
                        }
                        embedding_docs.append(embedding_doc)
                    
                    try:
                        if not embedding_docs:
                            logging.warning(f'No valid embeddings to store for {file_path}')
                            continue
                            
                        # Insert all embedding documents
                        result = collection.insert_many(embedding_docs)
                        successful += 1
                        logging.info(f'Successfully processed and stored {len(embedding_docs)} embeddings from {file_path}, MongoDB IDs: {len(result.inserted_ids)} documents')
                        
                        # Verify the inserts by counting the documents
                        inserted_count = collection.count_documents({
                            'filename': name,
                            'processed_at': base_data['processed_at']
                        })
                        
                        if inserted_count == len(embedding_docs):
                            logging.info(f'Verified {inserted_count} documents were inserted for {file_path}')
                        else:
                            logging.warning(f'Document count mismatch: expected {len(embedding_docs)}, found {inserted_count}')
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
