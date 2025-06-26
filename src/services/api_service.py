import os
import requests
import time
from typing import Optional, Dict, Any


class APIError(Exception):
    """Custom exception for API-related errors"""
    pass


class APIService:
    """API service class implementing Strategy pattern"""
    
    def __init__(self, url: str = None, max_retries: int = None, retry_delay: float = None):
        """Initialize with optional custom settings or load from environment variables"""
        # Get configuration from parameters or environment variables
        self.url = url or os.environ['API_URL']
        self.max_retries = max_retries or int(os.getenv('API_MAX_RETRIES', '3'))
        self.retry_delay = retry_delay or float(os.getenv('API_RETRY_DELAY', '2.0'))
        
        # Basic URL validation
        if not (self.url.startswith('https://') or self.url.startswith('http://')):
            raise ValueError("API URL must start with http:// or https://")
    
    def scan_image(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Scan an image using the API with retry mechanism"""
        # Use the validated URL
        url = self.url

        headers = {
            'Content-Type': 'multipart/form-data',
            'Accept': 'application/json'
        }

        for attempt in range(self.max_retries):
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                    response = requests.post(url, files=files, headers=headers, timeout=30)
                    if response.status_code != 200:
                        raise Exception(f"HTTP error: status {response.status_code}, body: {response.text}")
                    return response.json()
            except Exception as e:
                import logging
                logging.error(f"API error while processing {file_path}: {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
                import time
                time.sleep(self.retry_delay * (attempt + 1))
