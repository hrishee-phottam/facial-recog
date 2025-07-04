import io
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import boto3
import numpy as np
from PIL import Image
from botocore.exceptions import ClientError, BotoCoreError

from src.config import get_settings


class S3Error(Exception):
    """Custom exception for S3-related errors."""
    pass


class S3Service:
    """Service for S3/Wasabi storage operations."""
    
    def __init__(self, settings=None):
        """
        Initialize S3 service.
        
        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # S3 configuration
        self.bucket_name = self.settings.WASABI_BUCKET
        self.endpoint_url = self.settings.WASABI_ENDPOINT
        self.region = self.settings.WASABI_REGION
        
        # Initialize S3 client
        self.s3_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AWS S3/Wasabi client."""
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                region_name=self.region,
                aws_access_key_id=self.settings.WASABI_ACCESS_KEY,
                aws_secret_access_key=self.settings.WASABI_SECRET_KEY
            )
            
            # Test connection by listing bucket
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            self.logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
            
        except Exception as e:
            error_msg = f"Failed to initialize S3 client: {str(e)}"
            self.logger.error(error_msg)
            raise S3Error(error_msg) from e
    
    async def download_image(self, s3_path: str) -> Optional[bytes]:
        """
        Download image data from S3.
        
        Args:
            s3_path: S3 key/path to the image
            
        Returns:
            Optional[bytes]: Image data as bytes, or None if download fails
            
        Raises:
            S3Error: If download fails with unrecoverable error
        """
        if not self.s3_client:
            raise S3Error("S3 client not initialized")
        
        if not s3_path:
            raise S3Error("S3 path is required")
        
        try:
            self.logger.debug(f"Downloading image from S3: {s3_path}")
            
            # Use asyncio to run the synchronous S3 call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_path
                )
            )
            
            # Read the image data
            image_data = response['Body'].read()
            
            if not image_data:
                self.logger.warning(f"Empty image data received for {s3_path}")
                return None
            
            self.logger.debug(f"Successfully downloaded {len(image_data)} bytes from {s3_path}")
            return image_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'NoSuchKey':
                self.logger.warning(f"Image not found in S3: {s3_path}")
                return None
            elif error_code == 'NoSuchBucket':
                error_msg = f"S3 bucket not found: {self.bucket_name}"
                self.logger.error(error_msg)
                raise S3Error(error_msg) from e
            else:
                error_msg = f"S3 client error ({error_code}): {str(e)}"
                self.logger.error(error_msg)
                raise S3Error(error_msg) from e
                
        except BotoCoreError as e:
            error_msg = f"S3 service error: {str(e)}"
            self.logger.error(error_msg)
            raise S3Error(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error downloading from S3: {str(e)}"
            self.logger.error(error_msg)
            raise S3Error(error_msg) from e
    
    async def download_image_as_array(self, s3_path: str) -> Optional[np.ndarray]:
        """
        Download image from S3 and convert to numpy array.
        
        Args:
            s3_path: S3 key/path to the image
            
        Returns:
            Optional[np.ndarray]: Image as RGB numpy array, or None if conversion fails
        """
        try:
            # Download image data
            image_data = await self.download_image(s3_path)
            if image_data is None:
                return None
            
            # Convert to numpy array
            return self._bytes_to_array(image_data)
            
        except S3Error:
            # Re-raise S3Error as-is
            raise
        except Exception as e:
            error_msg = f"Error converting image to array for {s3_path}: {str(e)}"
            self.logger.error(error_msg)
            return None
    
    def _bytes_to_array(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Convert image bytes to RGB numpy array.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Optional[np.ndarray]: RGB numpy array, or None if conversion fails
        """
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            self.logger.debug(f"Converted image to array: {image_array.shape}")
            return image_array
            
        except Exception as e:
            self.logger.error(f"Error converting bytes to array: {str(e)}")
            return None
    
    # 🆕 NEW: Thumbnail upload functionality
    async def upload_thumbnail(self, thumbnail_data: bytes, s3_key: str) -> Tuple[bool, str]:
        """
        Upload thumbnail image to S3.
        
        Args:
            thumbnail_data: Thumbnail image as bytes
            s3_key: S3 key where to upload (e.g., "facetn/event_id/face_id.jpg")
            
        Returns:
            Tuple[bool, str]: (success, url) - success status and thumbnail URL
        """
        if not self.s3_client:
            raise S3Error("S3 client not initialized")
        
        if not thumbnail_data:
            self.logger.warning("Empty thumbnail data provided")
            return False, ""
        
        if not s3_key:
            self.logger.warning("Empty S3 key provided")
            return False, ""
        
        try:
            self.logger.debug(f"📸 Uploading thumbnail to S3: {s3_key}")
            
            # Use asyncio to run the synchronous S3 call
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=thumbnail_data,
                    ContentType='image/jpeg',
                    ACL='public-read'  # Make thumbnails publicly accessible
                )
            )
            
            # Generate public URL
            thumbnail_url = f"https://{self.bucket_name}.s3.{self.region}.wasabisys.com/{s3_key}"
            
            self.logger.info(f"✅ Thumbnail uploaded successfully: {s3_key}")
            return True, thumbnail_url
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"S3 client error uploading thumbnail ({error_code}): {str(e)}"
            self.logger.error(error_msg)
            return False, ""
            
        except BotoCoreError as e:
            error_msg = f"S3 service error uploading thumbnail: {str(e)}"
            self.logger.error(error_msg)
            return False, ""
            
        except Exception as e:
            error_msg = f"Unexpected error uploading thumbnail: {str(e)}"
            self.logger.error(error_msg)
            return False, ""
    
    def get_image_info(self, s3_path: str) -> Dict[str, Any]:
        """
        Get image metadata from S3.
        
        Args:
            s3_path: S3 key/path to the image
            
        Returns:
            Dict: Image metadata including size, last modified, etc.
        """
        try:
            if not self.s3_client:
                return {'error': 'S3 client not initialized'}
            
            # Get object metadata
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_path
            )
            
            metadata = {
                's3_path': s3_path,
                'size_bytes': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType', 'unknown'),
                'etag': response.get('ETag', '').strip('"')
            }
            
            return metadata
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            return {
                's3_path': s3_path,
                'error': f'S3 error ({error_code}): {str(e)}'
            }
        except Exception as e:
            return {
                's3_path': s3_path,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def extract_filename(self, s3_path: str) -> str:
        """
        Extract filename from S3 path.
        
        Args:
            s3_path: Full S3 path
            
        Returns:
            str: Filename without path
        """
        try:
            return Path(s3_path).name
        except Exception:
            return s3_path.split('/')[-1] if '/' in s3_path else s3_path
    
    def validate_image_path(self, s3_path: str) -> bool:
        """
        Validate if S3 path looks like an image file.
        
        Args:
            s3_path: S3 path to validate
            
        Returns:
            bool: True if path has image extension
        """
        try:
            valid_extensions = self.settings.SUPPORTED_EXTENSIONS
            path_lower = s3_path.lower()
            
            return any(path_lower.endswith(ext.lower()) for ext in valid_extensions)
            
        except Exception:
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get S3 service information.
        
        Returns:
            Dict: Service configuration and status
        """
        return {
            'bucket_name': self.bucket_name,
            'endpoint_url': self.endpoint_url,
            'region': self.region,
            'initialized': self.s3_client is not None,
            'supported_extensions': self.settings.SUPPORTED_EXTENSIONS
        }


# 🆕 NEW: Thumbnail creation utility
def create_thumbnail_from_bounding_box(image_array: np.ndarray, bounding_box: Dict[str, int], 
                                     thumbnail_size: Tuple[int, int] = (200, 200)) -> Optional[bytes]:
    """
    Create thumbnail from image using API bounding box coordinates.
    
    Args:
        image_array: Full image as RGB numpy array
        bounding_box: API bounding box with x_min, y_min, x_max, y_max
        thumbnail_size: Desired thumbnail size (width, height)
        
    Returns:
        Optional[bytes]: Thumbnail as JPEG bytes, or None if creation fails
    """
    try:
        # Extract coordinates from API bounding box
        x_min = int(bounding_box.get('x_min', 0))
        y_min = int(bounding_box.get('y_min', 0))
        x_max = int(bounding_box.get('x_max', 0))
        y_max = int(bounding_box.get('y_max', 0))
        
        # Validate coordinates
        img_height, img_width = image_array.shape[:2]
        if x_min >= x_max or y_min >= y_max or x_max > img_width or y_max > img_height:
            logging.warning(f"Invalid bounding box coordinates: {bounding_box}")
            return None
        
        # Calculate face dimensions
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        # Add smart padding (15% of face size)
        padding = max(10, int(min(face_width, face_height) * 0.15))
        
        # Apply padding with image boundary checks
        padded_x_min = max(0, x_min - padding)
        padded_y_min = max(0, y_min - padding)
        padded_x_max = min(img_width, x_max + padding)
        padded_y_max = min(img_height, y_max + padding)
        
        # Crop face with padding from original image
        face_crop = image_array[padded_y_min:padded_y_max, padded_x_min:padded_x_max]
        
        if face_crop.size == 0:
            logging.warning("Empty face crop resulted")
            return None
        
        # Convert to PIL Image
        face_pil = Image.fromarray(face_crop)
        
        # Resize to thumbnail size with high quality
        face_pil = face_pil.resize(thumbnail_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG bytes
        thumbnail_buffer = io.BytesIO()
        face_pil.save(thumbnail_buffer, format='JPEG', quality=90, optimize=True)
        thumbnail_bytes = thumbnail_buffer.getvalue()
        
        logging.debug(f"✅ Created thumbnail: {len(thumbnail_bytes)} bytes from box {bounding_box}")
        return thumbnail_bytes
        
    except Exception as e:
        logging.error(f"❌ Error creating thumbnail from bounding box: {str(e)}")
        return None


# Global S3 service instance
_s3_service = None


def get_s3_service() -> S3Service:
    """
    Get or create the global S3 service instance.
    
    Returns:
        S3Service: The S3 service instance
    """
    global _s3_service
    if _s3_service is None:
        _s3_service = S3Service()
    return _s3_service