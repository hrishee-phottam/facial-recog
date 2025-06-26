"""
Console observer for the face recognition system.

This module provides a ConsoleObserver class that displays processing progress
and results in the console in a user-friendly way.
"""
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import logging
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import processing result type for type hints
try:
    from .processor import ProcessingResult
except ImportError:
    # For when running this file directly
    from processor import ProcessingResult


class ConsoleObserver:
    """Displays processing progress and results in the console.
    
    This observer can be registered with the ImageProcessor to receive updates
    about the processing status of images.
    """
    
    def __init__(self, show_progress: bool = True, show_summary: bool = True):
        """Initialize the console observer.
        
        Args:
            show_progress: Whether to show a live progress bar.
            show_summary: Whether to show a summary of results.
        """
        self.console = Console()
        self.show_progress = show_progress
        self.show_summary = show_summary
        self.start_time = time.time()
        self.results: List[ProcessingResult] = []
        
        # Progress tracking
        self.processed = 0
        self.successful = 0
        self.failed = 0
    
    def __call__(
        self,
        file_path: str,
        result: Dict[str, Any],
        error: Optional[Exception] = None
    ) -> None:
        """Handle a processing event.
        
        Args:
            file_path: Path to the processed file.
            result: Result data from processing.
            error: Error that occurred during processing, if any.
        """
        self.processed += 1
        
        if error is not None:
            self.failed += 1
            self._display_error(file_path, error)
        else:
            self.successful += 1
            self._display_success(file_path, result)
    
    def _display_success(self, file_path: str, result: Dict[str, Any]) -> None:
        """Display a success message for a processed file."""
        filename = os.path.basename(file_path)
        face_count = len(result.get('result', []))
        
        status = Text(f"✓ {filename}", style="green")
        if face_count > 0:
            status.append(f" - Found {face_count} face{'s' if face_count != 1 else ''}", "dim")
        else:
            status.append(" - No faces detected", "yellow dim")
        
        self.console.print(status)
    
    def _display_error(self, file_path: str, error: Exception) -> None:
        """Display an error message for a failed file."""
        filename = os.path.basename(file_path)
        self.console.print(f"✗ {filename} - Error: {str(error)}", style="red")
    
    def display_summary(self) -> None:
        """Display a summary of the processing results."""
        if not self.show_summary or not self.processed:
            return
        
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Create a summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        table.add_row("Total files processed:", str(self.processed))
        table.add_row("Successful:", f"{self.successful} ({self.successful/self.processed*100:.1f}%)", 
                      style="green" if self.successful > 0 else "")
        table.add_row("Failed:", f"{self.failed} ({self.failed/self.processed*100:.1f}%)",
                     style="red" if self.failed > 0 else "")
        table.add_row("Processing time:", elapsed_str)
        
        self.console.print("\n" + "-" * 50)
        self.console.print(Panel.fit(
            table,
            title="[bold]Processing Complete[/bold]",
            border_style="blue"
        ))
    
    def get_progress_tracker(self, total: int):
        """Create and return a rich Progress instance for tracking progress.
        
        Args:
            total: Total number of items to process.
            
        Returns:
            A rich Progress instance configured for tracking.
        """
        return Progress(
            SpinnerColumn(),
            "• ",
            "[progress.description]{task.description}",
            "•",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            "[progress.completed]{task.completed}/{task.total}",
            "•",
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
