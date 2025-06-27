"""
Facial Recognition System - Main Entry Point

This module serves as the main entry point for the facial recognition system.
It orchestrates the image processing pipeline using the configured services.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv

# Import application components
from src.config import setup_logging, get_settings
from src.core.console_observer import ConsoleObserver
from src.core.processor import ImageProcessor


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Facial Recognition System - Process images for face detection and recognition.'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input directory containing images to process (overrides IMAGES_DIR from .env)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for processed results (not currently used)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=None,
        help='Set the logging level (overrides LOG_LEVEL from .env)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='facial_recognition.log',
        help='Path to the log file (default: facial_recognition.log)'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable the progress bar'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Disable the summary at the end of processing'
    )
    
    return parser.parse_args()


def validate_environment() -> None:
    """Validate that all required environment variables are set."""
    required_vars = [
        'API_URL',
        'MONGODB_URI',
        'MONGODB_DB_NAME',
        'MONGODB_COLLECTION_NAME',
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing)}")
        print("Please check your .env file and try again.")
        sys.exit(1)


def main() -> int:
    """
    Main function that orchestrates the facial recognition pipeline.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load environment from .env file
    load_dotenv('.env')
    
    # Validate required environment variables
    validate_environment()
    
    # Get settings with overrides from command line
    settings = get_settings()
    if args.log_level:
        settings.LOG_LEVEL = args.log_level
    
    # Configure logging
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        log_file=args.log_file
    )
    
    # Set up console observer
    console_observer = ConsoleObserver(
        show_progress=not args.no_progress,
        show_summary=not args.no_summary
    )
    print("👁️  Facial Recognition System - Starting up...")
    # Initialize the image processor
    processor = ImageProcessor(settings=settings)
    processor.register_observer(console_observer)
    
    # Determine input directory (command line takes precedence over .env)
    input_dir = args.input or settings.IMAGES_DIR
    try:
        # Process the directory
        console_observer.console.print(
            f"🔍 Starting face recognition processing in: {input_dir}",
            style="bold blue"
        )
        
        # Process all images in the directory
        results = processor.process_directory(input_dir)
        
        # Display summary
        console_observer.display_summary()
        
        # Return success if we processed at least one image successfully
        success_count = sum(1 for r in results if r.success)
        if success_count == 0 and results:
            return 1
        return 0
        
    except KeyboardInterrupt:
        console_observer.console.print("\n⚠️  Processing interrupted by user", style="yellow")
        return 130  # Standard exit code for Ctrl+C
        
    except Exception as e:
        console_observer.console.print(
            f"❌ Fatal error: {str(e)}",
            style="bold red"
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())
