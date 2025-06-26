import os
import requests
import time
from typing import Optional, Dict, Any
import ssl
from urllib3 import PoolManager

# Create a secure HTTPS connection pool
secure_pool = PoolManager(
    cert_reqs=ssl.CERT_REQUIRED,
    ca_certs=ssl.get_default_verify_paths().cafile
)


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
        
        # Validate API URL uses HTTPS
        if not self.url.startswith('https://'):
            raise ValueError("API URL must use HTTPS for secure communication")
    
    def scan_image(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Scan an image using the API with retry mechanism"""
        # Use the validated URL
        url = self.url

        headers = {
            'Content-Type': 'multipart/form-data',
            'Accept': 'application/json',
            'User-Agent': 'FaceRecog/1.0',
            'X-Request-ID': os.urandom(16).hex()  # Unique request ID for tracking
        }

        for attempt in range(self.max_retries):
            try:
                with open(file_path, 'rb') as f:
                    response = secure_pool.request(
                        'POST',
                        url,
                        fields={'file': (os.path.basename(file_path), f)},
                        headers=headers,
                        timeout=30  # 30 second timeout
                    )
                    if response.status != 200:
                        raise requests.exceptions.HTTPError(
                            f"HTTP {response.status}"
                        )
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
