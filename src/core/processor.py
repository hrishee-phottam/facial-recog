"""
Image processing module for the face recognition system.

This module provides the ImageProcessor class which handles the core image processing
pipeline, including API calls and database storage.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

# Import services
from src.services.api_service import APIService, APIError
from src.services.db_service import DBService, DBError
from src.config import get_settings

# Type aliases
ObserverCallback = Callable[[str, Dict[str, Any], Optional[Exception]], None]


@dataclass
class ProcessingResult:
    """Container for image processing results."""
    success: bool
    file_path: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageProcessor:
    """Orchestrates the image processing pipeline.
    
    Handles the entire process of:
    1. Scanning images using the API service
    2. Storing results in the database
    3. Notifying observers of processing events
    """
    
    def __init__(
        self,
        api_service: Optional[APIService] = None,
        db_service: Optional[DBService] = None,
        settings=None
    ):
        """Initialize the image processor.
        
        Args:
            api_service: Optional API service instance. If not provided, a new one will be created.
            db_service: Optional database service instance. If not provided, a new one will be created.
            settings: Optional settings instance. If not provided, global settings will be used.
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.api_service = api_service or APIService(settings=self.settings)
        self.db_service = db_service or DBService(settings=self.settings)
        
        # Observers for processing events
        self._observers: List[ObserverCallback] = []
    
    def register_observer(self, callback: ObserverCallback) -> None:
        """Register an observer callback to be notified of processing events.
        
        The callback should have the signature:
            callback(file_path: str, result: Dict[str, Any], error: Optional[Exception]) -> None
            
        Args:
            callback: Function to call when processing events occur.
        """
        if callback not in self._observers:
            self._observers.append(callback)
    
    def _notify_observers(
        self,
        file_path: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Notify all registered observers of a processing event.
        
        Args:
            file_path: Path to the file being processed.
            result: Optional result data from processing.
            error: Optional error that occurred during processing.
        """
        for callback in self._observers:
            try:
                callback(file_path, result or {}, error)
            except Exception as e:
                self.logger.error(
                    f"Error in observer callback {callback.__name__}: {str(e)}",
                    exc_info=True
                )
    
    def process_image(self, image_path: Union[str, Path]) -> ProcessingResult:
        """Process a single image file through the face recognition pipeline.
        
        Args:
            image_path: Path to the image file to process.
            
        Returns:
            ProcessingResult: Object containing the processing results.
        """
        result = ProcessingResult(
            success=False,
            file_path=str(image_path),
            metadata={
                'start_time': datetime.utcnow(),
                'end_time': None,
                'face_count': 0
            }
        )
        
        try:
            self.logger.debug(f"Starting processing: {image_path}")
            
            # 1. Call the API to scan the image
            try:
                api_result = self.api_service.scan_image(image_path)
                if not api_result or 'result' not in api_result:
                    raise ValueError("Invalid or empty API response")
                
                result.data = api_result
                result.metadata['face_count'] = len(api_result.get('result', []))
                self.logger.info(
                    f"API scan successful, found {result.metadata['face_count']} faces in {image_path}"
                )
                
            except APIError as e:
                raise RuntimeError(f"API error: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to process image with API: {str(e)}") from e
            
            # 2. Store results in database
            if result.metadata['face_count'] > 0:
                try:
                    for face in result.data['result']:
                        doc = {
                            'filename': os.path.basename(image_path),
                            'file_path': str(image_path),
                            'processed_at': datetime.utcnow(),
                            'embedding': face.get('embedding'),
                            'box': face.get('box', {}),
                            'execution_time': face.get('execution_time', {})
                        }
                        doc_id = self.db_service.store_result(doc)
                        self.logger.debug(f"Stored face data with ID: {doc_id}")
                    
                except DBError as e:
                    self.logger.error(f"Database error: {str(e)}")
                    # Don't fail the whole process if database storage fails
                except Exception as e:
                    self.logger.error(f"Unexpected error storing results: {str(e)}")
            
            # 3. Mark as successful
            result.success = True
            
        except Exception as e:
            result.error = e
            self.logger.error(
                f"Error processing {image_path}: {str(e)}",
                exc_info=True
            )
        
        # 4. Finalize metadata and notify observers
        result.metadata['end_time'] = datetime.utcnow()
        result.metadata['success'] = result.success
        
        # Notify observers
        if result.error:
            self._notify_observers(
                file_path=str(image_path),
                result=result.data,
                error=result.error
            )
        else:
            self._notify_observers(
                file_path=str(image_path),
                result=result.data
            )
        
        return result
    
    def process_directory(self, directory: Union[str, Path]) -> List[ProcessingResult]:
        """Process all supported images in a directory.
        
        Args:
            directory: Directory containing images to process.
            
        Returns:
            List of ProcessingResult objects for each processed file.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        results = []
        
        # Find all image files
        image_files = []
        for ext in self.settings.SUPPORTED_EXTENSIONS:
            image_files.extend(directory.rglob(f"*{ext}"))
        
        self.logger.info(f"Found {len(image_files)} images to process in {directory}")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            self.logger.info(f"Processing image {i}/{len(image_files)}: {image_path}")
            result = self.process_image(image_path)
            results.append(result)
            
            if result.error:
                self.logger.error(
                    f"Error processing {image_path}: {str(result.error)}"
                )
        
        return results
    
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
