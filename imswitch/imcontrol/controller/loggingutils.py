"""
CSV logging utilities for ImSwitch focus lock functionality.

Extracted from FocusLockController for better modularity and reusability.
Handles CSV file creation, rotation, and thread-safe logging.
"""

import os
import csv
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class CSVLogger:
    """
    Thread-safe CSV logger with automatic daily file rotation.
    
    Features:
    - Daily file rotation based on date
    - Thread-safe writing with locks
    - Automatic directory creation
    - Configurable field names and delimiter
    - Error handling and logging
    """

    def __init__(
        self,
        base_directory: str,
        file_prefix: str = "measurements",
        delimiter: str = ";",
        fieldnames: Optional[List[str]] = None,
    ):
        """
        Initialize CSV logger.
        
        Args:
            base_directory: Base directory for CSV files
            file_prefix: Prefix for CSV filenames
            delimiter: CSV delimiter character
            fieldnames: List of field names for CSV header
        """
        self.base_directory = base_directory
        self.file_prefix = file_prefix
        self.delimiter = delimiter
        self.fieldnames = fieldnames or []
        self._lock = threading.Lock()
        self._current_file: Optional[str] = None
        
        # Create directory if it doesn't exist
        self._ensure_directory_exists()
        
        logger.info(f"CSV logger initialized: {base_directory}")

    def _ensure_directory_exists(self) -> None:
        """Create base directory if it doesn't exist."""
        try:
            if not os.path.exists(self.base_directory):
                os.makedirs(self.base_directory)
                logger.info(f"Created CSV logging directory: {self.base_directory}")
        except Exception as e:
            logger.error(f"Failed to create CSV directory: {e}")
            raise

    def _get_current_filename(self) -> str:
        """Get filename for current date."""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.base_directory, f"{self.file_prefix}_{today}.csv")

    def _write_header_if_needed(self, filename: str) -> None:
        """Write CSV header if file is new."""
        if not os.path.exists(filename) and self.fieldnames:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames, delimiter=self.delimiter)
                    writer.writeheader()
                    logger.info(f"Created new CSV file with header: {filename}")
            except Exception as e:
                logger.error(f"Failed to write CSV header: {e}")
                raise

    def log_measurement(self, data: Dict[str, Any]) -> None:
        """
        Log a measurement to CSV file.
        
        Args:
            data: Dictionary containing measurement data
        """
        if not self.fieldnames:
            logger.warning("No fieldnames configured for CSV logger")
            return
            
        try:
            with self._lock:
                filename = self._get_current_filename()
                
                # Write header if this is a new file
                self._write_header_if_needed(filename)
                
                # Add timestamp and datetime if not present
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now().timestamp()
                if 'datetime' not in data:
                    data['datetime'] = datetime.fromtimestamp(data['timestamp']).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Write measurement data
                with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames, delimiter=self.delimiter)
                    
                    # Filter data to only include configured fieldnames
                    filtered_data = {field: data.get(field, None) for field in self.fieldnames}
                    writer.writerow(filtered_data)
                    
        except Exception as e:
            logger.error(f"Failed to log measurement to CSV: {e}")

    def log_focus_measurement(
        self,
        focus_value: float,
        timestamp: Optional[float] = None,
        is_locked: bool = False,
        lock_position: Optional[float] = None,
        current_position: Optional[float] = None,
        pi_output: Optional[float] = None,
        focus_metric: str = "unknown",
        crop_size: Optional[int] = None,
        crop_center: Optional[str] = None,
        **extra_fields
    ) -> None:
        """
        Log focus measurement with standard fields.
        
        Args:
            focus_value: Computed focus value
            timestamp: Measurement timestamp (auto-generated if None)
            is_locked: Whether focus lock is active
            lock_position: Z position when lock was engaged
            current_position: Current Z position
            pi_output: PID controller output
            focus_metric: Type of focus metric used
            crop_size: Image crop size
            crop_center: Image crop center coordinates
            **extra_fields: Additional fields to log
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        data = {
            'timestamp': timestamp,
            'focus_value': focus_value,
            'focus_metric': focus_metric,
            'is_locked': is_locked,
            'lock_position': lock_position,
            'current_position': current_position,
            'pi_output': pi_output,
            'crop_size': crop_size,
            'crop_center': crop_center,
            **extra_fields
        }
        
        self.log_measurement(data)

    def set_fieldnames(self, fieldnames: List[str]) -> None:
        """Update CSV fieldnames."""
        with self._lock:
            self.fieldnames = fieldnames
            logger.info(f"Updated CSV fieldnames: {fieldnames}")

    def get_current_file_path(self) -> str:
        """Get path to current CSV file."""
        return self._get_current_filename()

    def get_file_size(self) -> int:
        """Get size of current CSV file in bytes."""
        try:
            filename = self._get_current_filename()
            if os.path.exists(filename):
                return os.path.getsize(filename)
            return 0
        except Exception as e:
            logger.error(f"Failed to get CSV file size: {e}")
            return 0

    def get_record_count(self) -> int:
        """Get approximate number of records in current CSV file."""
        try:
            filename = self._get_current_filename()
            if not os.path.exists(filename):
                return 0
                
            with open(filename, 'r', encoding='utf-8') as csvfile:
                # Count lines and subtract 1 for header
                line_count = sum(1 for _ in csvfile)
                return max(0, line_count - 1)
        except Exception as e:
            logger.error(f"Failed to count CSV records: {e}")
            return 0

    def close(self) -> None:
        """Clean up resources (placeholder for future use)."""
        logger.info("CSV logger closed")


class FocusLockCSVLogger(CSVLogger):
    """
    Specialized CSV logger for focus lock measurements.
    
    Pre-configured with standard field names for focus lock data.
    """
    
    DEFAULT_FIELDNAMES = [
        'timestamp',
        'datetime', 
        'focus_value',
        'focus_metric',
        'is_locked',
        'lock_position',
        'current_position',
        'pi_output',
        'crop_size',
        'crop_center',
        'error',
        'integral_term',
        'derivative_term',
        'step_size_um',
        'travel_used_um',
    ]

    def __init__(self, base_directory: str, **kwargs):
        """Initialize focus lock CSV logger with default fieldnames."""
        super().__init__(
            base_directory=base_directory,
            file_prefix="focus_lock_measurements",
            fieldnames=kwargs.pop('fieldnames', self.DEFAULT_FIELDNAMES),
            **kwargs
        )

    def log_focus_lock_data(
        self,
        focus_value: float,
        is_locked: bool,
        current_position: float,
        timestamp: Optional[float] = None,
        pi_output: Optional[float] = None,
        error: Optional[float] = None,
        step_size_um: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log comprehensive focus lock data.
        
        Args:
            focus_value: Current focus metric value
            is_locked: Whether focus lock is engaged
            current_position: Current Z position
            timestamp: Measurement timestamp
            pi_output: PID controller output
            error: Current control error
            step_size_um: Step size in micrometers
            **kwargs: Additional fields
        """
        self.log_focus_measurement(
            focus_value=focus_value,
            timestamp=timestamp,
            is_locked=is_locked,
            current_position=current_position,
            pi_output=pi_output,
            error=error,
            step_size_um=step_size_um,
            **kwargs
        )