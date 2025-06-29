"""
API Service for face recognition operations.

This module provides the APIService class which handles all API interactions
with the face recognition service.
"""
import os
import time
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

import requests
from requests.exceptions import (
    RequestException, HTTPError, ConnectionError, Timeout, JSONDecodeError
)

from src.config import get_settings


class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


class APIService:
    """Service for interacting with the face recognition API."""
    
    def __init__(self, settings=None):
        """Initialize the API service with settings.
        
        Args:
            settings: Optional settings instance. If not provided, will use global settings.
            
        Raises:
            ValueError: If the API URL is invalid
        """
        self.settings = settings or get_settings()
        self.max_retries = self.settings.API_MAX_RETRIES
        self.retry_delay = self.settings.API_RETRY_DELAY
        
        # Process API URL
        self.base_url = str(self.settings.API_URL).rstrip('/')
        
        # Validate URL
        if not (self.base_url.startswith('https://') or self.base_url.startswith('http://')):
            raise ValueError("API URL must start with http:// or https://")
    
    def scan_image(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Scan an image file using the face recognition API.
        
        Args:
            file_path: Path to the image file to process.
            
        Returns:
            Dict containing the API response, or None if processing failed.
            
        Raises:
            ValueError: If the file doesn't exist or is not a file.
            APIError: If there's an error communicating with the API.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ValueError(f"File not found: {file_path}")
        
        # Ensure base URL doesn't have a trailing slash
        base_url = self.base_url.rstrip('/')
        
        # Use the base URL as is (already includes /scan_faces)
        url = base_url
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'FaceRecognition/1.0'
        }
        
        for attempt in range(self.max_retries):
            try:
                files = [
                    ('file', (file_path.name, open(str(file_path), 'rb'), 'application/octet-stream'))
                ]
                
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    timeout=self.settings.API_TIMEOUT
                )
                
                response.raise_for_status()
                return response.json()
                    
            except (ConnectionError, Timeout) as e:
                if attempt == self.max_retries - 1:
                    raise APIError(f"Failed to connect to API after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                
            except HTTPError as e:
                error_msg = f"API request failed with status {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f": {error_data.get('detail', str(error_data))}"
                except JSONDecodeError:
                    error_msg += f": {e.response.text}"
                raise APIError(error_msg) from e
                
            except RequestException as e:
                raise APIError(f"Request failed: {str(e)}") from e
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise APIError(f"Unexpected error: {str(e)}") from e
                time.sleep(self.retry_delay * (attempt + 1))
        
        return None
