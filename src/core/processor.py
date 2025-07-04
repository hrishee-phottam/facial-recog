import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import time

# Import System 2 services (keep unchanged)
from src.services.api_service import APIService, APIError
from src.services.db_service import DBService, DBError
from src.services.face_clustering_service import FaceClusteringService

# Import new production services
from src.services.sqs_service import SQSService, SQSError
from src.services.s3_service import S3Service, S3Error, create_thumbnail_from_bounding_box
from src.services.face_saver_service import FaceSaverService

from src.config import get_settings

# Type aliases
ObserverCallback = Callable[[str, Dict[str, Any], Optional[Exception]], None]


@dataclass
class ProcessingResult:
    """Container for image processing results (unchanged from System 2)."""
    success: bool
    file_path: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    clustering_result: Optional[Dict[str, Any]] = None


@dataclass
class SQSProcessingResult:
    """Container for SQS message processing results."""
    success: bool
    message_id: str
    media_id: str
    event_id: str
    s3_path: str
    faces_detected: int = 0
    faces_clustered: int = 0
    processing_time: float = 0.0
    error: Optional[Exception] = None
    ai_enabled: bool = True
    face_saver_result: Optional[Dict[str, Any]] = None
    thumbnails_uploaded: int = 0


class ImageProcessor:
    """
    Enhanced image processor with SQS workflow capabilities.
    
    Maintains all System 2 functionality while adding production SQS processing.
    """
    
    def __init__(
        self,
        api_service: Optional[APIService] = None,
        db_service: Optional[DBService] = None,
        clustering_service: Optional[FaceClusteringService] = None,
        sqs_service: Optional[SQSService] = None,
        s3_service: Optional[S3Service] = None,
        face_saver_service: Optional[FaceSaverService] = None,
        settings=None
    ):
        """Initialize the enhanced image processor."""
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize System 2 services (unchanged)
        self.api_service = api_service or APIService(settings=self.settings)
        self.db_service = db_service or DBService(settings=self.settings)
        
        # Initialize clustering service (optional)
        if self.settings.ENABLE_FACE_CLUSTERING:
            self.clustering_service = clustering_service or FaceClusteringService(self.db_service)
            self.logger.info(f"🧩 Face clustering enabled with threshold: {self.settings.SIMILARITY_THRESHOLD}")
        else:
            self.clustering_service = None
            self.logger.info("🧩 Face clustering disabled")
        
        # Initialize production services (new)
        self.sqs_service = sqs_service or SQSService(settings=self.settings)
        self.s3_service = s3_service or S3Service(settings=self.settings)
        self.face_saver_service = face_saver_service or FaceSaverService(settings=self.settings)
        
        # Observers for processing events (unchanged from System 2)
        self._observers: List[ObserverCallback] = []
        
        self.logger.info("🚀 Enhanced ImageProcessor initialized with production capabilities")
    
    # ==================== EXISTING SYSTEM 2 METHODS (UNCHANGED) ====================
    
    def register_observer(self, callback: ObserverCallback) -> None:
        """Register an observer callback to be notified of processing events."""
        if callback not in self._observers:
            self._observers.append(callback)
    
    def _notify_observers(
        self,
        file_path: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Notify all registered observers of a processing event."""
        for callback in self._observers:
            try:
                callback(file_path, result or {}, error)
            except Exception as e:
                self.logger.error(
                    f"❌ Error in observer callback {callback.__name__}: {str(e)}",
                    exc_info=True
                )
    
    def process_image(self, image_path: Union[str, Path]) -> ProcessingResult:
        """Process a single image file through the face recognition pipeline (unchanged)."""
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
            filename = os.path.basename(image_path)
            self.logger.info(f"📸 Processing image: {filename}")
            
            # 1. Call the API to scan the image
            try:
                self.logger.debug(f"🌐 Calling external API for face detection...")
                api_result = self.api_service.scan_image(image_path)
                if not api_result or 'result' not in api_result:
                    raise ValueError("Invalid or empty API response")
                
                result.data = api_result
                result.metadata['face_count'] = len(api_result.get('result', []))
                
                if result.metadata['face_count'] > 0:
                    self.logger.info(f"👥 API detected {result.metadata['face_count']} face(s) in {filename}")
                else:
                    self.logger.info(f"👤 No faces detected in {filename}")
                
            except APIError as e:
                raise RuntimeError(f"API error: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to process image with API: {str(e)}") from e
            
            # 2. Store results in database
            if result.metadata['face_count'] > 0:
                try:
                    for i, face in enumerate(result.data['result'], 1):
                        doc = {
                            'filename': filename,
                            'file_path': str(image_path),
                            'processed_at': datetime.utcnow(),
                            'embedding': face.get('embedding'),
                            'box': face.get('box', {}),
                            'execution_time': face.get('execution_time', {})
                        }
                        people_id = self.db_service.store_result(doc)
                        self.logger.debug(f"💾 Stored face {i} with ID: {people_id}")
                        
                        # Face clustering (optional)
                        if self.clustering_service and face.get('embedding'):
                            try:
                                clustering_result = self._cluster_face_sync(
                                    embedding=face.get('embedding'),
                                    people_id=people_id,
                                    filename=filename
                                )
                                result.clustering_result = clustering_result
                                
                                # Log clustering result
                                action = clustering_result.get('action', 'unknown')
                                if action == 'linked_existing':
                                    face_id = clustering_result.get('face_id')
                                    similarity = clustering_result.get('similarity_score', 0)
                                    self.logger.info(f"🎯 Face {i}: LINKED to {face_id} (similarity: {similarity:.3f})")
                                elif action == 'created_new':
                                    face_id = clustering_result.get('face_id')
                                    self.logger.info(f"🆕 Face {i}: NEW person created {face_id}")
                                elif action == 'grouped_similar':
                                    face_id = clustering_result.get('face_id')
                                    similarity = clustering_result.get('similarity_score', 0)
                                    self.logger.info(f"🔗 Face {i}: GROUPED into {face_id} (similarity: {similarity:.3f})")
                                
                            except Exception as cluster_error:
                                self.logger.warning(f"⚠️ Face {i} clustering failed: {str(cluster_error)}")
                    
                except DBError as e:
                    self.logger.error(f"❌ Database error: {str(e)}")
                except Exception as e:
                    self.logger.error(f"❌ Unexpected error storing results: {str(e)}")
            
            # 3. Mark as successful
            result.success = True
            self.logger.info(f"✅ Successfully processed {filename}")
            
        except Exception as e:
            result.error = e
            self.logger.error(f"❌ Error processing {image_path}: {str(e)}", exc_info=True)
        
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
        """Cluster a face embedding (sync version, unchanged from System 2)."""
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
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"❌ Face clustering error: {str(e)}")
            return {
                'action': 'error',
                'face_id': None,
                'similarity_score': 0.0,
                'is_new_face': False,
                'error': str(e)
            }
    
    def process_directory(self, directory: Union[str, Path]) -> List[ProcessingResult]:
        """Process all supported images in a directory (unchanged from System 2)."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        results = []
        
        # File deduplication
        self.logger.info(f"🔍 Scanning for images in {directory}")
        image_files_set = set()
        
        for ext in self.settings.SUPPORTED_EXTENSIONS:
            found_files = directory.rglob(f"*{ext}")
            for file_path in found_files:
                resolved_path = file_path.resolve()
                image_files_set.add(resolved_path)
        
        image_files = sorted(list(image_files_set))
        self.logger.info(f"📂 Found {len(image_files)} unique images to process")
        
        # Track processed files
        processed_in_run = set()
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            resolved_path = image_path.resolve()
            if resolved_path in processed_in_run:
                self.logger.info(f"⏭️ Skipping {image_path.name} - already processed in this run")
                continue
            
            processed_in_run.add(resolved_path)
            
            self.logger.info(f"📸 Processing image {i}/{len(image_files)}: {image_path.name}")
            result = self.process_image(image_path)
            results.append(result)
            
            if result.error:
                self.logger.error(f"❌ Error processing {image_path}: {str(result.error)}")
        
        # Log clustering summary if enabled
        if self.clustering_service:
            try:
                summary = self.clustering_service.get_clustering_summary()
                self.logger.info(f"🧩 Final clustering summary: {summary['unique_faces']} unique faces, "
                               f"{summary['clustering_rate']} clustering rate")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not get clustering summary: {str(e)}")
        
        return results
    
    def get_clustering_stats(self) -> Optional[Dict[str, Any]]:
        """Get face clustering statistics (unchanged from System 2)."""
        if self.clustering_service:
            return self.clustering_service.get_clustering_summary()
        return None
    
    # ==================== NEW ENHANCED SQS METHODS WITH CLEAN LOGGING ====================
    
    def log_image_header(self, filename: str, media_id: str, event_id: str, s3_path: str, event_name: str = "Unknown"):
        """Log clean image processing header"""
        self.logger.info("┌─ 📥 SQS MESSAGE: {} ─────────────────────────────────────────────".format(filename))
        self.logger.info(f"│ Media: {media_id} | Event: {event_id}")
        self.logger.info(f"│ S3: {s3_path}")
        self.logger.info("└─ AI Status: ✅ ENABLED (Event: '{}') ──────────────────────────────────".format(event_name))
    
    def log_detection_complete(self, faces_detected: int, s3_time: float, api_time: float, save_path: str, faces_saved: int):
        """Log face detection completion"""
        self.logger.info("🔍 FACE DETECTION COMPLETE ──────────────────────────────────────────────────")
        self.logger.info(f"   📊 API Response: {faces_detected} faces detected | S3 Download: {s3_time:.1f}s | API Call: {api_time:.1f}s")
        if faces_saved > 0:
            self.logger.info(f"   💾 Local Save: {faces_saved}/{faces_detected} faces saved → {save_path}")
        self.logger.info(f"   ⏱️  Phase Time: {s3_time + api_time:.1f}s")
    
    def log_image_complete(self, filename: str, media_id: str, total_time: float, faces_detected: int, 
                          faces_clustered: int, thumbnails_uploaded: int, faces_new: int, faces_matched: int,
                          detection_time: float, clustering_time: float, finalization_time: float, face_results: List):
        """Log image processing completion"""
        self.logger.info("✅ IMAGE PROCESSING COMPLETE ────────────────────────────────────────────────")
        self.logger.info(f"   📸 File: {filename} | Media: {media_id} | Total Time: {total_time:.1f}s")
        self.logger.info("")
        self.logger.info("   📊 RESULTS SUMMARY:")
        
        if faces_new > 0 and faces_matched > 0:
            result_text = f"{faces_detected} detected, {faces_new} new + {faces_matched} matched persons"
        elif faces_new > 0:
            result_text = f"{faces_detected} detected, {faces_new} new persons created"
        else:
            result_text = f"{faces_detected} detected, {faces_matched} matched to existing persons"
        
        self.logger.info(f"   ┌─ Faces: {result_text} ──────────────────────────────")
        
        # Log individual face results
        for i, face_result in enumerate(face_results, 1):
            if face_result.get('action') == 'new':
                closest = face_result.get('closest_similarity', 0)
                face_id_short = face_result.get('face_id', '')[-8:] if face_result.get('face_id') else 'unknown'
                self.logger.info(f"   │  👤 Face {i} → ID: ...{face_id_short} (closest existing: {closest:.3f})")
            else:
                similarity = face_result.get('similarity', 0)
                face_id_short = face_result.get('face_id', '')[-8:] if face_result.get('face_id') else 'unknown'
                source_file = face_result.get('source_file', 'unknown')
                self.logger.info(f"   │  👤 Face {i} → MATCHED ID: ...{face_id_short} (similarity: {similarity:.3f}) ← {source_file}")
        
        self.logger.info(f"   ├─ Thumbnails: {thumbnails_uploaded}/{faces_detected} uploaded successfully ─────────────────────────────")
        
        if self.face_saver_service.is_enabled():
            self.logger.info(f"   ├─ Local Saves: {faces_detected}/{faces_detected} faces saved locally ──────────────────────────────")
        
        self.logger.info(f"   └─ Database: Media updated with {faces_clustered} face references ────────────────────────")
        self.logger.info(f"   ⏱️  Breakdown: Detection({detection_time:.1f}s) + Clustering({clustering_time:.1f}s) + Finalization({finalization_time:.1f}s)")
    
    def log_image_separator(self):
        """Log separator between images"""
        self.logger.info("")
        self.logger.info("════════════════════════════════════════════════════════════════════════════════")
        self.logger.info("")
    
    async def process_sqs_message(self, message: Dict[str, Any]) -> SQSProcessingResult:
        """
        Process a single SQS message through the complete face recognition workflow.
        
        Args:
            message: Raw SQS message
            
        Returns:
            SQSProcessingResult: Complete processing results
        """
        start_time = datetime.utcnow()
        
        try:
            # Parse message
            try:
                parsed_data = self.sqs_service.parse_message(message)
            except SQSError as e:
                return SQSProcessingResult(
                    success=False,
                    message_id=message.get('MessageId', 'unknown'),
                    media_id='unknown',
                    event_id='unknown',
                    s3_path='unknown',
                    error=e
                )
            
            media_id = parsed_data['mediaId']
            s3_path = parsed_data['path']
            event_id = parsed_data['eventId']
            message_id = parsed_data['messageId']
            filename = os.path.basename(s3_path)
            
            # Check if AI is enabled for this event
            ai_enabled = await self.db_service.check_event_ai_enabled(event_id)
            
            if not ai_enabled:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                self.logger.info(f"⏭️ AI disabled for event - skipping processing ({processing_time:.1f}s)")
                return SQSProcessingResult(
                    success=True,
                    message_id=message_id,
                    media_id=media_id,
                    event_id=event_id,
                    s3_path=s3_path,
                    ai_enabled=False,
                    processing_time=processing_time
                )
            
            # Update media status to pending
            await self.db_service.update_media_status(media_id, 'p')
            
            # Process media from S3
            processing_result = await self.process_media_from_s3(media_id, s3_path, event_id)
            
            # Update media status based on result
            if processing_result.success:
                await self.db_service.update_media_status(media_id, 'c')
            else:
                await self.db_service.update_media_status(media_id, 'f')
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing SQS message: {str(e)}", exc_info=True)
            
            # Try to update media status to failed if we have the info
            try:
                if 'media_id' in locals():
                    await self.db_service.update_media_status(media_id, 'f')
            except:
                pass
            
            return SQSProcessingResult(
                success=False,
                message_id=message.get('MessageId', 'unknown'),
                media_id=locals().get('media_id', 'unknown'),
                event_id=locals().get('event_id', 'unknown'),
                s3_path=locals().get('s3_path', 'unknown'),
                error=e,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def process_media_from_s3(self, media_id: str, s3_path: str, event_id: str) -> SQSProcessingResult:
        """
        Process a media file from S3 through the face recognition workflow.
        
        Args:
            media_id: Media ID
            s3_path: S3 path to the image
            event_id: Event ID
            
        Returns:
            SQSProcessingResult: Processing results
        """
        start_time = datetime.utcnow()
        filename = os.path.basename(s3_path)
        
        # Timing variables
        s3_download_time = 0
        api_call_time = 0
        detection_start = time.time()
        
        try:
            # Get event name for logging
            event_name = "Unknown"
            try:
                event = await self.db_service.events_collection.find_one(
                    {"_id": event_id},
                    {"name": 1}
                )
                if event:
                    event_name = event.get('name', 'Unknown')
            except:
                pass
            
            # Log clean image header
            self.log_image_header(filename, media_id, event_id, s3_path, event_name)
            
            # Extract orgId from S3 path
            org_id = None
            try:
                path_parts = s3_path.split('/')
                if len(path_parts) >= 2 and path_parts[0] == 'media':
                    org_id = path_parts[1]
            except Exception:
                pass
            
            # Validate image path
            if not self.s3_service.validate_image_path(s3_path):
                raise ValueError(f"Invalid image path: {s3_path}")
            
            # Download image from S3
            try:
                s3_start = time.time()
                image_array = await self.s3_service.download_image_as_array(s3_path)
                s3_download_time = time.time() - s3_start
                
                if image_array is None:
                    raise ValueError("Failed to download or convert image")
                
            except S3Error as e:
                raise RuntimeError(f"S3 download error: {str(e)}") from e
            
            # ✅ CHANGE 2: Process with external API - handle zero faces as normal
            try:
                import tempfile
                import cv2
                
                api_start = time.time()
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(tmp_file.name, image_bgr)
                    
                    api_result = self.api_service.scan_image(tmp_file.name)
                    os.unlink(tmp_file.name)
                
                api_call_time = time.time() - api_start
                
                if not api_result or 'result' not in api_result:
                    raise ValueError("Invalid or empty API response")
                
                faces_data = api_result.get('result', [])
                faces_detected = len(faces_data)
                
                # ✅ CHANGE 3: Log zero faces as INFO, not ERROR
                if faces_detected == 0:
                    self.logger.info(f"👤 No faces detected in {filename} (normal)")
                
            except APIError as e:
                # ✅ CHANGE 4: Don't wrap APIError in RuntimeError - handle directly
                self.logger.error(f"❌ API error for {filename}: {str(e)}")
                raise e
            
            # Save faces locally if enabled
            face_saver_result = None
            faces_saved = 0
            save_path = ""
            
            if self.face_saver_service.is_enabled() and faces_data:
                try:
                    formatted_faces = []
                    for face in faces_data:
                        box = face.get('box', {})
                        if box:
                            face_location = (
                                box.get('y_min', 0),
                                box.get('x_max', 0),
                                box.get('y_max', 0),
                                box.get('x_min', 0)
                            )
                            formatted_faces.append({
                                'face_location': face_location,
                                'box': box,
                                'quality_score': 0.8
                            })
                    
                    face_saver_result = self.face_saver_service.save_detected_faces(
                        image_array, formatted_faces, s3_path, event_id
                    )
                    
                    if face_saver_result:
                        faces_saved = face_saver_result.get('faces_saved', 0)
                        save_path = face_saver_result.get('save_directory', '')
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Face saving failed: {str(e)}")
            
            detection_time = time.time() - detection_start
            
            # Log detection completion
            self.log_detection_complete(faces_detected, s3_download_time, api_call_time, save_path, faces_saved)
            
            # Store results and perform clustering
            face_ids = []
            faces_clustered = 0
            thumbnails_uploaded = 0
            faces_new = 0
            faces_matched = 0
            face_results = []
            
            clustering_start = time.time()
            
            if faces_detected > 0:
                for i, face in enumerate(faces_data, 1):
                    try:
                        face_start = time.time()
                        
                        # Store face in people collection with media reference
                        doc = {
                            'filename': filename,
                            'file_path': s3_path,
                            'processed_at': datetime.utcnow(),
                            'embedding': face.get('embedding'),
                            'box': face.get('box', {}),
                            'execution_time': face.get('execution_time', {})
                        }
                        people_id = self.db_service.store_result_with_media(doc, media_id)
                        
                        face_result = {'face_index': i}
                        
                        # Perform clustering if enabled
                        if self.clustering_service and face.get('embedding'):
                            try:
                                clustering_result = await self.clustering_service.process_new_embedding(
                                    embedding=face.get('embedding'),
                                    people_id=people_id,
                                    metadata={
                                        'filename': filename,
                                        'eventId': event_id,
                                        'orgId': org_id
                                    }
                                )
                                
                                face_id = clustering_result.get('face_id')
                                if face_id:
                                    face_ids.append(face_id)
                                    faces_clustered += 1
                                    
                                    # Create and upload thumbnail
                                    await self._create_and_upload_thumbnail(
                                        image_array, face, face_id, event_id, i
                                    )
                                    thumbnails_uploaded += 1
                                    
                                    # Track face results for summary
                                    action = clustering_result.get('action', 'unknown')
                                    if action == 'linked_existing':
                                        faces_matched += 1
                                        face_result.update({
                                            'action': 'matched',
                                            'face_id': face_id,
                                            'similarity': clustering_result.get('similarity_score', 0),
                                            'source_file': 'Previous image'  # Could be enhanced to show actual source
                                        })
                                    else:
                                        faces_new += 1
                                        face_result.update({
                                            'action': 'new',
                                            'face_id': face_id,
                                            'closest_similarity': clustering_result.get('closest_similarity', 0)
                                        })
                                
                                face_time = time.time() - face_start
                                
                            except Exception as cluster_error:
                                self.logger.warning(f"⚠️ Clustering failed for face {i}: {str(cluster_error)}")
                        
                        face_results.append(face_result)
                        
                    except Exception as face_error:
                        self.logger.error(f"❌ Error processing face {i}: {str(face_error)}")
                
                # Update media with face references
                if face_ids:
                    try:
                        await self.db_service.update_media_with_faces(media_id, face_ids)
                    except Exception as e:
                        self.logger.error(f"❌ Error updating media with faces: {str(e)}")
            
            clustering_time = time.time() - clustering_start
            finalization_start = time.time()
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            finalization_time = time.time() - finalization_start
            
            # Log comprehensive completion
            self.log_image_complete(
                filename, media_id, processing_time, faces_detected, faces_clustered, 
                thumbnails_uploaded, faces_new, faces_matched, detection_time, 
                clustering_time, finalization_time, face_results
            )
            
            return SQSProcessingResult(
                success=True,
                message_id='processed',
                media_id=media_id,
                event_id=event_id,
                s3_path=s3_path,
                faces_detected=faces_detected,
                faces_clustered=faces_clustered,
                processing_time=processing_time,
                ai_enabled=True,
                face_saver_result=face_saver_result,
                thumbnails_uploaded=thumbnails_uploaded
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"❌ FAILED: {filename} - {str(e)} ({processing_time:.1f}s)", exc_info=True)
            
            return SQSProcessingResult(
                success=False,
                message_id='error',
                media_id=media_id,
                event_id=event_id,
                s3_path=s3_path,
                error=e,
                processing_time=processing_time
            )
    
    async def _create_and_upload_thumbnail(self, image_array, face_data, face_id, event_id, face_index):
        """
        Create and upload thumbnail for a face.
        
        Args:
            image_array: Full image as numpy array
            face_data: Face data from API (contains bounding box)
            face_id: Face ID from clustering
            event_id: Event ID
            face_index: Face index for logging
        """
        try:
            # Get bounding box from API response
            bounding_box = face_data.get('box', {})
            if not bounding_box:
                return
            
            # Create thumbnail from bounding box
            thumbnail_data = create_thumbnail_from_bounding_box(image_array, bounding_box)
            if not thumbnail_data:
                return
            
            # Generate S3 key for thumbnail
            thumbnail_key = f"facetn/{event_id}/{face_id}.jpg"
            
            # Upload thumbnail to S3
            success, thumbnail_url = await self.s3_service.upload_thumbnail(thumbnail_data, thumbnail_key)
            
            if success:
                # Update face record with tScore
                face_width = bounding_box.get('x_max', 0) - bounding_box.get('x_min', 0)
                face_height = bounding_box.get('y_max', 0) - bounding_box.get('y_min', 0)
                face_area = face_width * face_height
                
                if face_area > 10000:
                    tScore = 0.9
                elif face_area > 5000:
                    tScore = 0.8
                else:
                    tScore = 0.7
                
                try:
                    from bson import ObjectId
                    self.db_service.faces_collection.update_one(
                        {"_id": ObjectId(face_id)},
                        {"$set": {"tScore": tScore}}
                    )
                except Exception:
                    pass
                
        except Exception as e:
            self.logger.error(f"❌ Face {face_index}: Thumbnail creation/upload error: {str(e)}")
    
    async def run_sqs_processor(self, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the SQS message processing loop.
        
        Args:
            max_iterations: Maximum number of polling iterations (None for infinite)
            
        Returns:
            Dict: Processing statistics
        """
        stats = {
            'messages_processed': 0,
            'messages_successful': 0,
            'messages_failed': 0,
            'messages_skipped_ai_disabled': 0,
            'total_faces_detected': 0,
            'total_faces_clustered': 0,
            'total_thumbnails_uploaded': 0,
            'start_time': datetime.utcnow(),
            'end_time': None
        }
        
        iterations = 0
        
        try:
            self.logger.info("🔄 Starting SQS message processing loop...")
            
            while True:
                if max_iterations and iterations >= max_iterations:
                    break
                
                try:
                    # Receive messages from SQS
                    messages = await self.sqs_service.receive_messages()
                    
                    if not messages:
                        await asyncio.sleep(1)
                        continue
                    
                    self.logger.info(f"📥 Processing SQS messages...")
                    self.logger.info("")
                    
                    # Process each message
                    for message in messages:
                        try:
                            # Process the message
                            result = await self.process_sqs_message(message)
                            
                            stats['messages_processed'] += 1
                            
                            if result.success:
                                if result.ai_enabled:
                                    stats['messages_successful'] += 1
                                    stats['total_faces_detected'] += result.faces_detected
                                    stats['total_faces_clustered'] += result.faces_clustered
                                    stats['total_thumbnails_uploaded'] += getattr(result, 'thumbnails_uploaded', 0)
                                else:
                                    stats['messages_skipped_ai_disabled'] += 1
                                
                                # Delete message from queue
                                await self.sqs_service.delete_message(message['ReceiptHandle'])
                                
                            else:
                                stats['messages_failed'] += 1
                                
                                # Delete message to prevent infinite retries
                                await self.sqs_service.delete_message(message['ReceiptHandle'])
                            
                            # Add separator after each image
                            self.log_image_separator()
                            
                        except Exception as message_error:
                            stats['messages_failed'] += 1
                            self.logger.error(f"❌ Error processing message: {str(message_error)}", exc_info=True)
                            
                            # Delete problematic message
                            try:
                                await self.sqs_service.delete_message(message['ReceiptHandle'])
                            except:
                                pass
                
                except SQSError as e:
                    self.logger.error(f"❌ SQS error: {str(e)}")
                    await asyncio.sleep(5)
                
                except Exception as e:
                    self.logger.error(f"❌ Unexpected error in SQS loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(5)
                
                iterations += 1
            
        except KeyboardInterrupt:
            self.logger.info("🛑 SQS processing interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Fatal error in SQS processor: {str(e)}", exc_info=True)
        finally:
            stats['end_time'] = datetime.utcnow()
            stats['total_processing_time'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        return stats