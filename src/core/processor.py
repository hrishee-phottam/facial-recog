import logging
import os
from typing import List, Dict, Any
from datetime import datetime
from src.services.api_service import APIService
from src.services.db_service import DBService

class ImageProcessor:
    """Main image processing class implementing Observer pattern"""
    
    def __init__(self):
        # Initialize with environment variables
        self.image_dir = os.getenv('IMAGES_DIR', 'images')
        self.supported_extensions = tuple(os.getenv('SUPPORTED_EXTENSIONS', '.jpg,.jpeg,.png,.bmp,.gif').split(','))
        self.api_service = APIService()
        self.db_service = DBService()
        self._observers = []
        
    def register_observer(self, observer):
        """Register an observer to be notified of processing events"""
        self._observers.append(observer)
    
    def notify_observers(self, event_type: str, data: Dict[str, Any]):
        """Notify all registered observers of an event"""
        for observer in self._observers:
            observer.update(event_type, data)
    
    def process_images(self, directory: str = None):
        """Process all images in the specified directory"""
        directory = directory or self.image_dir
        
        try:
            # Count total images
            total_images = self._count_images(directory)
            if total_images == 0:
                logging.info("No images found in directory")
                return
            
            logging.info(f"Found {total_images} images to process")
            
            processed = 0
            successful = 0
            failed = 0
            
            # Process each image
            for root, _, files in os.walk(directory):
                for name in files:
                    if name.lower().endswith(self.config.SUPPORTED_EXTENSIONS):
                        file_path = os.path.join(root, name)
                        try:
                            self._process_single_image(file_path)
                            successful += 1
                        except Exception as e:
                            failed += 1
                            logging.error(f"Failed to process {file_path}: {str(e)}")
                        finally:
                            processed += 1
                            
                            # Notify observers of progress
                            progress = {
                                'processed': processed,
                                'total': total_images,
                                'successful': successful,
                                'failed': failed
                            }
                            self.notify_observers('progress', progress)
                            
                            # Log progress periodically
                            if processed % self.config.BATCH_SIZE == 0:
                                self._log_progress(processed, total_images, successful, failed)
            
            # Final summary
            self._log_summary(processed, successful, failed)
            
        except Exception as e:
            logging.error(f"Critical error in processing: {str(e)}")
            self.notify_observers('error', {'message': str(e)})
    
    def _count_images(self, directory: str) -> int:
        """Count total number of images in directory"""
        return sum(1 for root, _, files in os.walk(directory)
                  for name in files if name.lower().endswith(self.config.SUPPORTED_EXTENSIONS))
    
    def _process_single_image(self, file_path: str):
        """Process a single image file"""
        logging.info(f"Scanning {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        logging.info(f"File size: {file_size/1024/1024:.2f} MB")
        
        # Scan the image
        data = self.api_service.scan_image(file_path)
        if data is None:
            raise Exception(f"Failed to process {file_path}")
        
        # Add metadata
        data['filename'] = os.path.basename(file_path)
        data['processed_at'] = datetime.now().isoformat()
        data['file_size'] = file_size
        
        # Store in database
        mongo_id = self.db_service.store_result(data)
        logging.info(f"Successfully processed and stored {file_path}, MongoDB ID: {mongo_id}")
        
    def _log_progress(self, processed: int, total: int, successful: int, failed: int):
        """Log progress periodically"""
        logging.info(f"\nProgress: {processed}/{total} processed")
        logging.info(f"Success rate: {successful/processed*100:.2f}%")
        logging.info(f"Successful: {successful}, Failed: {failed}")
    
    def _log_summary(self, processed: int, successful: int, failed: int):
        """Log final summary"""
        logging.info('\n=== Processing Summary ===')
        logging.info(f'Total images processed: {processed}')
        logging.info(f'Successful scans: {successful}')
        logging.info(f'Failed scans: {failed}')
        logging.info(f'Success rate: {successful/processed*100:.2f}%')
        logging.info('=== End of Processing ===')
