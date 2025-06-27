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
import asyncio

# Import services
from src.services.api_service import APIService, APIError
from src.services.db_service import DBService, DBError
from src.services.face_clustering_service import FaceClusteringService  # 🆕 NEW
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
    clustering_result: Optional[Dict[str, Any]] = None  # 🆕 NEW


class ImageProcessor:
    """Orchestrates the image processing pipeline.
    
    Handles the entire process of:
    1. Scanning images using the API service
    2. Storing results in the database
    3. Clustering faces (optional)
    4. Notifying observers of processing events
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
        
        # 🆕 NEW: Initialize clustering service (optional)
        if self.settings.ENABLE_FACE_CLUSTERING:
            self.clustering_service = FaceClusteringService(self.db_service)
            self.logger.info(f"Face clustering enabled with threshold: {self.settings.SIMILARITY_THRESHOLD}")
        else:
            self.clustering_service = None
            self.logger.info("Face clustering disabled")
        
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
                        people_id = self.db_service.store_result(doc)
                        self.logger.debug(f"Stored face data with ID: {people_id}")
                        
                        # 🆕 NEW: Face clustering (optional)
                        if self.clustering_service and face.get('embedding'):
                            try:
                                clustering_result = self._cluster_face_sync(
                                    embedding=face.get('embedding'),
                                    people_id=people_id,
                                    filename=os.path.basename(image_path)
                                )
                                result.clustering_result = clustering_result
                                
                            except Exception as cluster_error:
                                self.logger.warning(f"Face clustering failed: {str(cluster_error)}")
                    
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
    
    def _cluster_face_sync(self, embedding: List[float], people_id: str, filename: str) -> Dict[str, Any]:
        """
        🆕 NEW: Cluster a face embedding (sync version).
        
        Args:
            embedding: Face embedding vector
            people_id: ID of the stored embedding
            filename: Source filename
            
        Returns:
            Clustering result dictionary
        """
        try:
            metadata = {'filename': filename}
            
            # Run async clustering in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                clustering_result = loop.run_until_complete(
                    self.clustering_service.process_new_embedding(
                        embedding=embedding,
                        people_id=people_id,
                        metadata=metadata
                    )
                )
            finally:
                loop.close()
            
            # Log clustering result
            action = clustering_result.get('action', 'unknown')
            if action == 'linked_existing':
                face_id = clustering_result.get('face_id')
                similarity = clustering_result.get('similarity_score', 0)
                self.logger.info(f"🎯 Face linked to existing person {face_id} (similarity: {similarity:.3f})")
            elif action == 'created_new':
                face_id = clustering_result.get('face_id')
                self.logger.info(f"🆕 New unique person created: {face_id}")
            elif action == 'grouped_similar':
                face_id = clustering_result.get('face_id')
                similarity = clustering_result.get('similarity_score', 0)
                self.logger.info(f"🔗 Grouped with similar face into {face_id} (similarity: {similarity:.3f})")
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"Face clustering error: {str(e)}")
            return {
                'action': 'error',
                'face_id': None,
                'similarity_score': 0.0,
                'is_new_face': False,
                'error': str(e)
            }

    async def _cluster_face(self, embedding: List[float], people_id: str, filename: str) -> Dict[str, Any]:
        """
        🆕 NEW: Cluster a face embedding.
        
        Args:
            embedding: Face embedding vector
            people_id: ID of the stored embedding
            filename: Source filename
            
        Returns:
            Clustering result dictionary
        """
        try:
            metadata = {'filename': filename}
            clustering_result = await self.clustering_service.process_new_embedding(
                embedding=embedding,
                people_id=people_id,
                metadata=metadata
            )
            
            # Log clustering result
            action = clustering_result.get('action', 'unknown')
            if action == 'linked_existing':
                face_id = clustering_result.get('face_id')
                similarity = clustering_result.get('similarity_score', 0)
                self.logger.info(f"🎯 Face linked to existing person {face_id} (similarity: {similarity:.3f})")
            elif action == 'created_new':
                face_id = clustering_result.get('face_id')
                self.logger.info(f"🆕 New unique person created: {face_id}")
            elif action == 'grouped_similar':
                face_id = clustering_result.get('face_id')
                similarity = clustering_result.get('similarity_score', 0)
                self.logger.info(f"🔗 Grouped with similar face into {face_id} (similarity: {similarity:.3f})")
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"Face clustering error: {str(e)}")
            return {
                'action': 'error',
                'face_id': None,
                'similarity_score': 0.0,
                'is_new_face': False,
                'error': str(e)
            }
    
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
        
        # 🆕 NEW: File deduplication to prevent same file processing multiple times
        self.logger.info(f"Scanning for images in {directory}")
        image_files_set = set()
        
        for ext in self.settings.SUPPORTED_EXTENSIONS:
            found_files = directory.rglob(f"*{ext}")
            for file_path in found_files:
                # Use resolved path to handle symlinks and duplicates
                resolved_path = file_path.resolve()
                image_files_set.add(resolved_path)
        
        # Convert to sorted list
        image_files = sorted(list(image_files_set))
        
        self.logger.info(f"Found {len(image_files)} unique images to process")
        
        # Track processed files within this run to catch any remaining duplicates
        processed_in_run = set()
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            # Final duplicate check within this run
            resolved_path = image_path.resolve()
            if resolved_path in processed_in_run:
                self.logger.info(f"⏭️ Skipping {image_path.name} - already processed in this run")
                continue
            
            processed_in_run.add(resolved_path)
            
            self.logger.info(f"Processing image {i}/{len(image_files)}: {image_path}")
            result = self.process_image(image_path)
            results.append(result)
            
            if result.error:
                self.logger.error(
                    f"Error processing {image_path}: {str(result.error)}"
                )
        
        # 🆕 NEW: Log clustering summary if enabled
        if self.clustering_service:
            try:
                summary = self.clustering_service.get_clustering_summary()
                self.logger.info(f"🧩 Clustering Summary: {summary['unique_faces']} unique faces, "
                               f"{summary['clustering_rate']} clustering rate")
            except Exception as e:
                self.logger.warning(f"Could not get clustering summary: {str(e)}")
        
        return results
    
    def get_clustering_stats(self) -> Optional[Dict[str, Any]]:
        """
        🆕 NEW: Get face clustering statistics.
        
        Returns:
            Clustering statistics or None if clustering disabled
        """
        if self.clustering_service:
            return self.clustering_service.get_clustering_summary()
        return None
    
    # Existing methods below remain unchanged
    
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
        
        # 🆕 NEW: Log clustering stats if enabled
        if self.clustering_service:
            try:
                clustering_stats = self.get_clustering_stats()
                if clustering_stats:
                    logging.info(f"🧩 Unique faces: {clustering_stats.get('unique_faces', 0)}")
                    logging.info(f"🔗 Clustering rate: {clustering_stats.get('clustering_rate', '0.0%')}")
            except Exception as e:
                logging.warning(f"Could not retrieve clustering stats: {str(e)}")
        
        logging.info('=== 🏁 End of Processing ===')