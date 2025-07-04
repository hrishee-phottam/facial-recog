import os
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
from pydantic import HttpUrl, validator, Field, field_validator, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main settings class that contains all configuration - flattened for Pydantic compatibility"""
    
    # API Configuration
    API_URL: HttpUrl = Field(..., description="Base URL for the face recognition API")
    API_MAX_RETRIES: int = Field(3, description="Maximum number of retries for API calls")
    API_RETRY_DELAY: float = Field(2.0, description="Delay between retries in seconds")
    API_TIMEOUT: int = Field(90, description="Timeout for API requests in seconds")
    
    # MongoDB Configuration
    MONGODB_URI: str = Field(..., description="MongoDB connection URI")
    MONGODB_DB_NAME: str = Field("dev", description="Name of the MongoDB database")
    MONGODB_COLLECTION_NAME: str = Field("people", description="Name of the people collection")
    MONGODB_CONNECT_TIMEOUT_MS: int = Field(5000, description="MongoDB connection timeout in milliseconds")
    MONGODB_SERVER_SELECTION_TIMEOUT_MS: int = Field(10000, description="MongoDB server selection timeout in milliseconds")
    
    # SQS Configuration
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., description="AWS secret access key")
    AWS_REGION: str = Field("ap-southeast-1", description="AWS region")
    SQS_QUEUE_URL: str = Field(..., description="SQS queue URL")
    SQS_POLL_WAIT_TIME: int = Field(20, description="SQS long polling wait time")
    SQS_MAX_MESSAGES: int = Field(1, description="Maximum messages to receive per poll")
    SQS_VISIBILITY_TIMEOUT: int = Field(600, description="Message visibility timeout in seconds")
    PROCESSING_TIMEOUT_MINUTES: int = Field(30, description="Processing timeout per message")
    
    # S3/Wasabi Configuration
    WASABI_BUCKET: str = Field(..., description="Wasabi/S3 bucket name")
    WASABI_ACCESS_KEY: str = Field(..., description="Wasabi access key")
    WASABI_SECRET_KEY: str = Field(..., description="Wasabi secret key")
    WASABI_ENDPOINT: str = Field(..., description="Wasabi endpoint URL")
    WASABI_REGION: str = Field("ap-northeast-1", description="Wasabi region")
    
    # Face Clustering Configuration
    ENABLE_FACE_CLUSTERING: bool = Field(True, description="Enable face clustering feature")
    SIMILARITY_THRESHOLD: float = Field(0.83, description="Face similarity threshold for clustering")
    VECTOR_INDEX_NAME: str = Field("face_embedding_index", description="MongoDB vector search index name")
    FACES_COLLECTION_NAME: str = Field("faces", description="Collection name for unique faces")
    
    # Face Saver Configuration
    SAVE_DETECTED_FACES_LOCALLY: bool = Field(False, description="Enable local face saving for debugging")
    LOCAL_FACES_DIR: str = Field("./detected_faces", description="Directory to save detected faces")
    FACE_QUALITY_THRESHOLD: float = Field(0.0, description="Minimum quality threshold for saving faces")
    
    # Image Processing Configuration
    IMAGES_DIR: Union[DirectoryPath, str] = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "images"),
        description="Directory containing images to process"
    )
    SUPPORTED_EXTENSIONS: Union[str, List[str]] = Field(
        ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.webp', '.WEBP'],
        description="Comma-separated string or list of supported image file extensions"
    )
    MAX_IMAGE_SIZE_MB: int = Field(10, description="Maximum image size in MB")
    
    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    LOG_FILE: str = Field("facial_recognition.log", description="Path to the log file")
    
    # General Configuration
    ENVIRONMENT: str = Field("development", description="Application environment")
    DEBUG: bool = Field(False, description="Debug mode flag")
    SHOW_PROGRESS: bool = Field(True, description="Whether to show progress bars")
    SHOW_SUMMARY: bool = Field(True, description="Whether to show summary after processing")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        validate_assignment=True,
        extra='ignore'  # Ignore extra fields in environment variables
    )
    
    @field_validator('MONGODB_URI', mode='before')
    @classmethod
    def validate_mongodb_uri(cls, v):
        """Validate MongoDB URI format."""
        if not v:
            raise ValueError("MONGODB_URI is required")
        if not v.startswith(('mongodb://', 'mongodb+srv://')):
            raise ValueError("MONGODB_URI must start with 'mongodb://' or 'mongodb+srv://'")
        return v
    
    @field_validator('SIMILARITY_THRESHOLD')
    @classmethod
    def validate_similarity_threshold(cls, v):
        """Ensure similarity threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('SIMILARITY_THRESHOLD must be between 0.0 and 1.0')
        return v
    
    @field_validator('SUPPORTED_EXTENSIONS', mode='before')
    @classmethod
    def validate_extensions(cls, v):
        """Ensure all extensions start with a dot and are lowercase."""
        if not v:
            return ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            
        # Handle both string and list inputs
        if isinstance(v, str):
            # Split comma-separated string and strip whitespace
            extensions = [ext.strip() for ext in v.split(',')]
        elif isinstance(v, list):
            extensions = v
        else:
            raise ValueError("SUPPORTED_EXTENSIONS must be a comma-separated string or list")
            
        # Ensure all extensions start with a dot
        return [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions if ext]
    
    @field_validator('IMAGES_DIR', mode='before')
    @classmethod
    def validate_images_dir(cls, v):
        """Ensure the images directory exists or can be created."""
        if not v:
            v = os.path.join(os.getcwd(), "images")
        
        # Convert to absolute path
        v = os.path.abspath(v)
        
        # Create directory if it doesn't exist
        os.makedirs(v, exist_ok=True)
        
        return v
    
    @field_validator('LOCAL_FACES_DIR', mode='before')
    @classmethod
    def validate_faces_dir(cls, v):
        """Ensure the faces directory can be created."""
        if not v:
            v = "./detected_faces"
        
        # Convert to absolute path
        v = os.path.abspath(v)
        
        return v
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    @property
    def mongodb_connection_params(self) -> dict:
        """Get MongoDB connection parameters."""
        return {
            'host': self.MONGODB_URI,
            'connectTimeoutMS': self.MONGODB_CONNECT_TIMEOUT_MS,
            'serverSelectionTimeoutMS': self.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
            'retryWrites': True,
            'w': 'majority'
        }
    
    @property
    def supported_extensions_str(self) -> str:
        """Get supported file extensions as a comma-separated string."""
        return ', '.join(self.SUPPORTED_EXTENSIONS)
    
    def validate_required_settings(self) -> bool:
        """Validate that all required settings for production are present"""
        required_settings = [
            self.API_URL,
            self.MONGODB_URI,
            self.AWS_ACCESS_KEY_ID,
            self.AWS_SECRET_ACCESS_KEY,
            self.SQS_QUEUE_URL,
            self.WASABI_BUCKET,
            self.WASABI_ACCESS_KEY,
            self.WASABI_SECRET_KEY,
            self.WASABI_ENDPOINT,
        ]
        
        missing_settings = []
        for setting in required_settings:
            if not setting:
                missing_settings.append(str(setting))
        
        if missing_settings:
            print(f"Missing required environment variables: {missing_settings}")
            return False
        
        return True


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """
    Get or create the global settings instance.
    
    Returns:
        Settings: The application settings
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def initialize_settings() -> Settings:
    """
    Initialize and validate settings for production use.
    
    Returns:
        Settings: Validated settings instance
        
    Raises:
        Exception: If required settings are missing
    """
    settings = get_settings()
    
    if not settings.validate_required_settings():
        raise Exception("Missing required environment variables for production")
    
    return settings