"""
Logging configuration for the facial recognition system.

This module sets up logging with colored output and file logging.
"""
import logging
import sys
from typing import Optional
from pathlib import Path

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

def setup_logging(
    log_level: str = 'INFO',
    log_format: str = '%(asctime)s - [%(levelname)s] - %(message)s',
    log_file: Optional[str] = 'facial_recognition.log',
    log_to_console: bool = True
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        log_file: Path to log file. If None, file logging is disabled
        log_to_console: Whether to log to console
    """
    # Convert string log level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    colored_formatter = ColoredFormatter(log_format)
    
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
    
    # Set log level for third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
