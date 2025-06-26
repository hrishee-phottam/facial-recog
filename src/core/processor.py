import logging
import os
from typing import List, Dict, Any
from datetime import datetime
from src.services.api_service import APIService
from src.services.db_service import DBService
import json

# Colored formatter for logging with emojis
class ColoredFormatter(logging.Formatter):
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

# Set up colored logging globally (will only affect new handlers)
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_format = os.getenv('LOG_FORMAT', '%(asctime)s - [%(levelname)s] - %(message)s')
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[logging.StreamHandler(), logging.FileHandler('scan_and_store.log')]
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logging.getLogger().handlers[0] = console_handler

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
                    if name.lower().endswith(self.supported_extensions):
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
                            # Log progress every 10 images by default
                            if processed % 10 == 0:
                                self._log_progress(processed, total_images, successful, failed)
            
            # Final summary
            self._log_summary(processed, successful, failed)
            
        except Exception as e:
            logging.error(f"Critical error in processing: {str(e)}")
            self.notify_observers('error', {'message': str(e)})
    
    def _count_images(self, directory: str) -> int:
        """Count total number of images in directory"""
        return sum(1 for root, _, files in os.walk(directory)
                  for name in files if name.lower().endswith(self.supported_extensions))
    
    def _process_single_image(self, file_path: str):
        """Process a single image file with detailed logging and verification"""
        logging.info(f"📸 Scanning {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        logging.info(f"📁 File size: {file_size/1024/1024:.2f} MB")
        
        # Scan the image
        data = self.api_service.scan_image(file_path)
        if data is None:
            raise Exception(f"Failed to process {file_path}")
        
        # Add metadata
        data['filename'] = os.path.basename(file_path)
        data['processed_at'] = datetime.now().isoformat()
        data['file_size'] = file_size

        # If multiple faces/embeddings, store each as a separate doc (with coordinates)
        face_embeddings = []
        if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
            face_index = 0
            for result in data['result']:
                if isinstance(result, dict) and 'embedding' in result and result['embedding']:
                    face_box = result.get('box', {})
                    logging.debug(f"Processing face {face_index}: box={face_box}")
                    if face_box and all(k in face_box for k in ['x_min', 'x_max', 'y_min', 'y_max']):
                        logging.debug(f"Found valid box coordinates: {face_box}")
                        face_doc = {
                            'embedding': result['embedding'],
                            'embedding_length': len(result['embedding']),
                            'embedding_index': face_index,
                            'face_count': len(data.get('result', [])),
                            'file_path': file_path,
                            'filename': data['filename'],
                            'processed_at': data['processed_at'],
                            'execution_time': result.get('execution_time', {}),
                            'box': {
                                'x_min': face_box['x_min'],
                                'x_max': face_box['x_max'],
                                'y_min': face_box['y_min'],
                                'y_max': face_box['y_max'],
                            }
                        }
                        face_embeddings.append(face_doc)
                    face_index += 1
                else:
                    logging.warning(f"Skipping face {face_index} due to missing box coordinates: {face_box.keys() if 'face_box' in locals() else 'N/A'}")
                    continue
        
        if not face_embeddings:
            logging.warning(f'No face embeddings found in {file_path}')
            return
        logging.info(f'Found {len(face_embeddings)} face(s) in {file_path}')
        
        # Insert all embeddings at once and verify
        try:
            logging.debug(f"Attempting to insert {len(face_embeddings)} documents")
            logging.debug(f"First document sample: {json.dumps(face_embeddings[0], default=str)}")
            result = self.db_service.collection.insert_many(face_embeddings)
            logging.info(f"Successfully stored {len(face_embeddings)} face embeddings for {data['filename']}")
            logging.debug(f"MongoDB insert result: {result.inserted_ids}")

            # Verify immediately after insertion
            inserted_ids = result.inserted_ids
            if inserted_ids:
                logging.debug(f"Verifying documents with IDs: {inserted_ids}")
                found_docs = list(self.db_service.collection.find({'_id': {'$in': inserted_ids}}))
                logging.debug(f"Found {len(found_docs)} documents after immediate verification")
                if found_docs:
                    logging.debug(f"Sample document: {json.dumps(found_docs[0], default=str)}")
        except Exception as e:
            logging.error(f"Error storing embeddings for {data['filename']}: {str(e)}")
            raise

    def _log_progress(self, processed: int, total: int, successful: int, failed: int):
        """Log progress periodically"""
        logging.info(f"\nProgress: {processed}/{total} processed")
        logging.info(f"Success rate: {successful/processed*100:.2f}%")
        logging.info(f"Successful: {successful}, Failed: {failed}")
    
    def _log_summary(self, processed: int, successful: int, failed: int):
        """Log final summary with emojis and enhanced stats"""
        logging.info('\n=== 📝 Processing Summary ===')
        logging.info(f'Total images processed: {processed}')
        logging.info(f'✅ Successful scans: {successful}')
        logging.info(f'❌ Failed scans: {failed}')
        if processed > 0:
            logging.info(f'📊 Success rate: {successful/processed*100:.2f}%')
        else:
            logging.info('No images processed.')
        # Optionally, log DB stats
        try:
            stats = self.db_service.get_stats()
            logging.info(f"🗄️ DB total documents: {stats.get('total_documents')}")
            logging.info(f"🕑 Last processed: {stats.get('last_processed')}")
        except Exception as e:
            logging.warning(f"Could not retrieve DB stats: {str(e)}")
        logging.info('=== 🏁 End of Processing ===')
