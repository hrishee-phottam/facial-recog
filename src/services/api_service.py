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
        """Initialize with optional custom settings"""
        # Get configuration from environment variables
        self.url = url or os.getenv('API_URL', 'http://47.129.240.165:3000/scan_faces')
        self.max_retries = max_retries or int(os.getenv('API_MAX_RETRIES', '3'))
        self.retry_delay = retry_delay or float(os.getenv('API_RETRY_DELAY', '2.0'))
    
    def scan_image(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Scan an image using the API with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                with open(file_path, 'rb') as f:
                    response = requests.post(self.url, files={'file': f}, timeout=None)
                    response.raise_for_status()
                    return response.json()
            except requests.exceptions.ConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise APIError(f"Failed to connect after {self.max_retries} attempts for {file_path}")
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
            except requests.exceptions.Timeout:
                raise APIError(f"Request timed out for {file_path}")
            except requests.exceptions.HTTPError as e:
                raise APIError(f"HTTP error {e.response.status_code} for {file_path}")
            except Exception as e:
                raise APIError(f"Unexpected error while processing {file_path}: {str(e)}")
        return None
