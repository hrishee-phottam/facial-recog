import logging
import sys
import os
from typing import Optional
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output with emoji support"""
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[37m',     # White
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET)
        return f'{color}{log_message}{self.RESET}'

def setup_logging(
    log_level: str = 'INFO',
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
    log_file: Optional[str] = 'facial_recognition.log',
    log_to_console: bool = True,
    batch_number: Optional[int] = None
) -> None:
    """
    Configure logging for the application with System 2's clean style.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        log_file: Path to log file. If None, file logging is disabled
        log_to_console: Whether to log to console
        batch_number: Optional batch number for organized logging
    """
    # Convert string log level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # ✅ NEW: Handle batch logging structure
    if batch_number and log_file:
        # Create batch folder structure: facial_recognition_logs/1/facial_recognition.log
        base_log_dir = "facial_recognition_logs"
        batch_dir = os.path.join(base_log_dir, str(batch_number))
        
        # Ensure batch directory exists
        os.makedirs(batch_dir, exist_ok=True)
        
        # Update log file path to use batch directory
        log_file = os.path.join(batch_dir, "facial_recognition.log")
    
    # Create formatters
    # CLEAN format for console (NO TIMESTAMPS - just the message)
    console_format = '%(message)s'
    # Detailed format for file (WITH timestamps for debugging)
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(file_format)
    colored_formatter = ColoredFormatter(console_format)  # Clean console format
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(colored_formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    # Set log level for third-party libraries to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Log setup completion
    logger.info("🔧 Logging system initialized")
    logger.info(f"   Log level: {log_level}")
    logger.info(f"   Console output: {'✅ Enabled' if log_to_console else '❌ Disabled'}")
    logger.info(f"   File logging: {'✅ Enabled' if log_file else '❌ Disabled'}")
    if log_file:
        logger.info(f"   Log file: {log_file}")
    if batch_number:
        logger.info(f"   Batch number: {batch_number}")