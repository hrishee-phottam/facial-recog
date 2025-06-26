import logging
from typing import Dict, Any

class ConsoleObserver:
    """Observer that displays processing status in the console"""
    
    def update(self, event_type: str, data: Dict[str, Any]):
        """Handle update events from the processor"""
        if event_type == 'progress':
            self._display_progress(data)
        elif event_type == 'error':
            logging.error(f"Error: {data['message']}")
    
    def _display_progress(self, progress: Dict[str, Any]):
        """Display progress information in the console"""
        processed = progress['processed']
        total = progress['total']
        successful = progress['successful']
        failed = progress['failed']
        
        print(f"\rProgress: {processed}/{total} processed", end='')
        print(f" | Success rate: {successful/processed*100:.1f}%", end='')
        print(f" | Successful: {successful} | Failed: {failed}", end='')
