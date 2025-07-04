import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
from PIL import Image

from src.config import get_settings


class FaceSaverError(Exception):
    """Custom exception for face saver-related errors."""
    pass


class FaceSaverService:
    """Service for saving detected faces locally for debugging."""
    
    def __init__(self, settings=None):
        """
        Initialize face saver service.
        
        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Face saver configuration
        self.enabled = self.settings.SAVE_DETECTED_FACES_LOCALLY
        self.base_dir = self.settings.LOCAL_FACES_DIR
        self.quality_threshold = self.settings.FACE_QUALITY_THRESHOLD
        
        # Statistics
        self.stats = {
            'total_faces_processed': 0,
            'faces_saved': 0,
            'faces_skipped_quality': 0,
            'faces_skipped_disabled': 0,
            'save_errors': 0
        }
        
        if self.enabled:
            self._ensure_base_directory()
            self.logger.info(f"Face saving enabled: {self.base_dir} (quality threshold: {self.quality_threshold})")
        else:
            self.logger.info("Face saving disabled")
    
    def _ensure_base_directory(self):
        """Ensure base directory exists."""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            self.logger.debug(f"Ensured base directory exists: {self.base_dir}")
        except Exception as e:
            error_msg = f"Failed to create base directory {self.base_dir}: {str(e)}"
            self.logger.error(error_msg)
            raise FaceSaverError(error_msg) from e
    
    def is_enabled(self) -> bool:
        """
        Check if face saving is enabled.
        
        Returns:
            bool: True if face saving is enabled
        """
        return self.enabled
    
    def save_detected_faces(self, image_array: np.ndarray, faces_data: List[Dict[str, Any]], 
                          s3_path: str, event_id: str) -> Dict[str, Any]:
        """
        Save all detected faces from an image.
        
        Args:
            image_array: Original image as numpy array
            faces_data: List of face detection results
            s3_path: S3 path of the original image
            event_id: Event ID for organizing saves
            
        Returns:
            Dict: Save operation results
        """
        if not self.enabled:
            self.stats['faces_skipped_disabled'] += len(faces_data)
            return {
                'enabled': False,
                'faces_processed': len(faces_data),
                'faces_saved': 0,
                'message': 'Face saving disabled'
            }
        
        if not faces_data:
            return {
                'enabled': True,
                'faces_processed': 0,
                'faces_saved': 0,
                'message': 'No faces to save'
            }
        
        try:
            # Extract filename from S3 path
            filename = self._extract_filename(s3_path)
            
            # Create directory structure: {base_dir}/{event_id}/{filename}/
            save_dir = self._get_save_directory(event_id, filename)
            
            saved_count = 0
            skipped_count = 0
            errors = []
            
            # Save each face
            for face_index, face_data in enumerate(faces_data):
                try:
                    self.stats['total_faces_processed'] += 1
                    
                    # Check quality threshold
                    quality_score = face_data.get('quality_score', 0.0)
                    if quality_score < self.quality_threshold:
                        skipped_count += 1
                        self.stats['faces_skipped_quality'] += 1
                        self.logger.debug(f"Skipped face {face_index + 1} due to quality: {quality_score:.3f} < {self.quality_threshold}")
                        continue
                    
                    # Save individual face
                    face_saved = self._save_single_face(
                        image_array, face_data, save_dir, face_index, quality_score
                    )
                    
                    if face_saved:
                        saved_count += 1
                        self.stats['faces_saved'] += 1
                    else:
                        skipped_count += 1
                        self.stats['save_errors'] += 1
                    
                except Exception as e:
                    error_msg = f"Error saving face {face_index + 1}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
                    self.stats['save_errors'] += 1
            
            # Create summary file
            if saved_count > 0:
                self._create_summary_file(save_dir, s3_path, event_id, faces_data, saved_count)
            
            result = {
                'enabled': True,
                'faces_processed': len(faces_data),
                'faces_saved': saved_count,
                'faces_skipped': skipped_count,
                'save_directory': save_dir,
                'quality_threshold': self.quality_threshold
            }
            
            if errors:
                result['errors'] = errors
            
            if saved_count > 0:
                self.logger.info(f"Saved {saved_count}/{len(faces_data)} faces to {save_dir}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in save_detected_faces: {str(e)}"
            self.logger.error(error_msg)
            return {
                'enabled': True,
                'faces_processed': len(faces_data),
                'faces_saved': 0,
                'error': error_msg
            }
    
    def _save_single_face(self, image_array: np.ndarray, face_data: Dict[str, Any], 
                         save_dir: str, face_index: int, quality_score: float) -> bool:
        """
        Save a single face crop.
        
        Args:
            image_array: Original image array
            face_data: Face detection data
            save_dir: Directory to save face
            face_index: Index of this face
            quality_score: Quality score of this face
            
        Returns:
            bool: True if face was saved successfully
        """
        try:
            # Extract face location from face_data
            face_location = face_data.get('face_location')
            if not face_location:
                # Try to get from box data
                box = face_data.get('box', {})
                if box:
                    # Convert box format to face_location format (top, right, bottom, left)
                    top = box.get('y_min', 0)
                    right = box.get('x_max', 0)
                    bottom = box.get('y_max', 0)
                    left = box.get('x_min', 0)
                    face_location = (top, right, bottom, left)
                else:
                    self.logger.warning(f"No face location data for face {face_index + 1}")
                    return False
            
            # Extract face region
            top, right, bottom, left = face_location
            face_crop = image_array[top:bottom, left:right]
            
            if face_crop.size == 0:
                self.logger.warning(f"Empty face crop for face {face_index + 1}")
                return False
            
            # Convert to PIL Image
            face_pil = Image.fromarray(face_crop)
            
            # Generate filename with quality info
            quality_str = f"{quality_score:.3f}".replace('.', '_')
            face_filename = f"face_{face_index + 1:02d}_q{quality_str}.jpg"
            face_path = os.path.join(save_dir, face_filename)
            
            # Save with high quality
            face_pil.save(face_path, format='JPEG', quality=95, optimize=True)
            
            self.logger.debug(f"Saved face {face_index + 1}: {face_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving face {face_index + 1}: {str(e)}")
            return False
    
    def _get_save_directory(self, event_id: str, filename: str) -> str:
        """
        Get save directory for faces.
        
        Args:
            event_id: Event ID
            filename: Original filename
            
        Returns:
            str: Directory path for saving faces
        """
        # Create directory structure: {base_dir}/{event_id}/{filename_without_ext}/
        filename_base = Path(filename).stem
        save_dir = os.path.join(self.base_dir, event_id, filename_base)
        
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        return save_dir
    
    def _extract_filename(self, s3_path: str) -> str:
        """
        Extract filename from S3 path.
        
        Args:
            s3_path: Full S3 path
            
        Returns:
            str: Filename only
        """
        try:
            return Path(s3_path).name
        except Exception:
            return s3_path.split('/')[-1] if '/' in s3_path else s3_path
    
    def _create_summary_file(self, save_dir: str, s3_path: str, event_id: str, 
                           faces_data: List[Dict[str, Any]], saved_count: int):
        """
        Create a summary file with processing information.
        
        Args:
            save_dir: Directory where faces were saved
            s3_path: Original S3 path
            event_id: Event ID
            faces_data: Face detection data
            saved_count: Number of faces actually saved
        """
        try:
            summary_path = os.path.join(save_dir, '_summary.txt')
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Face Extraction Summary\n")
                f.write(f"======================\n\n")
                f.write(f"Original Image: {s3_path}\n")
                f.write(f"Event ID: {event_id}\n")
                f.write(f"Processed At: {datetime.now().isoformat()}\n")
                f.write(f"Total Faces Detected: {len(faces_data)}\n")
                f.write(f"Faces Saved: {saved_count}\n")
                f.write(f"Quality Threshold: {self.quality_threshold}\n\n")
                
                f.write(f"Face Details:\n")
                f.write(f"-------------\n")
                
                for i, face_data in enumerate(faces_data):
                    quality = face_data.get('quality_score', 0.0)
                    saved = quality >= self.quality_threshold
                    f.write(f"Face {i + 1:2d}: Quality {quality:.3f} - {'SAVED' if saved else 'SKIPPED'}\n")
            
            self.logger.debug(f"Created summary file: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating summary file: {str(e)}")
    
    def get_save_path(self, event_id: str, filename: str) -> str:
        """
        Get the save path for a given event and filename.
        
        Args:
            event_id: Event ID
            filename: Original filename
            
        Returns:
            str: Full save directory path
        """
        return self._get_save_directory(event_id, filename)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get face saving statistics.
        
        Returns:
            Dict: Statistics about face saving operations
        """
        total_processed = self.stats['total_faces_processed']
        
        stats = {
            'enabled': self.enabled,
            'base_directory': self.base_dir,
            'quality_threshold': self.quality_threshold,
            **self.stats
        }
        
        if total_processed > 0:
            stats['save_rate'] = (self.stats['faces_saved'] / total_processed) * 100
        else:
            stats['save_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'total_faces_processed': 0,
            'faces_saved': 0,
            'faces_skipped_quality': 0,
            'faces_skipped_disabled': 0,
            'save_errors': 0
        }
        self.logger.info("Face saver statistics reset")
    
    def cleanup_old_saves(self, days_old: int = 7) -> Dict[str, Any]:
        """
        Clean up old saved faces.
        
        Args:
            days_old: Remove faces older than this many days
            
        Returns:
            Dict: Cleanup results
        """
        if not self.enabled:
            return {'enabled': False, 'message': 'Face saving disabled'}
        
        try:
            from datetime import timedelta
            import time
            
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            removed_count = 0
            removed_dirs = []
            
            if os.path.exists(self.base_dir):
                for event_dir in os.listdir(self.base_dir):
                    event_path = os.path.join(self.base_dir, event_dir)
                    if os.path.isdir(event_path):
                        # Check if directory is old enough
                        dir_mtime = os.path.getmtime(event_path)
                        if dir_mtime < cutoff_time:
                            import shutil
                            shutil.rmtree(event_path)
                            removed_dirs.append(event_dir)
                            
                            # Count files that would have been removed
                            for root, dirs, files in os.walk(event_path):
                                removed_count += len([f for f in files if f.endswith('.jpg')])
            
            result = {
                'enabled': True,
                'days_old': days_old,
                'directories_removed': len(removed_dirs),
                'approximate_files_removed': removed_count,
                'removed_directories': removed_dirs
            }
            
            if removed_dirs:
                self.logger.info(f"Cleaned up {len(removed_dirs)} old face directories")
            
            return result
            
        except Exception as e:
            error_msg = f"Error cleaning up old saves: {str(e)}"
            self.logger.error(error_msg)
            return {'enabled': True, 'error': error_msg}


# Global face saver service instance
_face_saver_service = None


def get_face_saver_service() -> FaceSaverService:
    """
    Get or create the global face saver service instance.
    
    Returns:
        FaceSaverService: The face saver service instance
    """
    global _face_saver_service
    if _face_saver_service is None:
        _face_saver_service = FaceSaverService()
    return _face_saver_service