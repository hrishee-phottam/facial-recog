import logging
import os
from src.core.processor import ImageProcessor
from src.core.console_observer import ConsoleObserver


def setup_logging():
    """Setup logging configuration using environment variables"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function"""
    try:
        # Setup logging
        setup_logging()
        
        # Initialize processor and observer
        processor = ImageProcessor()
        observer = ConsoleObserver()
        
        # Register observer
        processor.register_observer(observer)
        
        # Process images
        processor.process_images()
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
