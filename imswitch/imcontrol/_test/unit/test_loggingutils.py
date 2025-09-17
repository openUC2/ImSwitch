"""
Unit tests for CSV logging utilities.
"""

import pytest
import os
import csv
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch

from imswitch.imcontrol.controller.loggingutils import CSVLogger, FocusLockCSVLogger


class TestCSVLogger:
    """Test the CSV logging utility."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = CSVLogger(
            base_directory=self.test_dir,
            file_prefix="test",
            fieldnames=['timestamp', 'datetime', 'value', 'status']
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_csv_logger_initialization(self):
        """Test CSV logger initialization."""
        assert self.logger.base_directory == self.test_dir
        assert self.logger.file_prefix == "test"
        assert self.logger.delimiter == ";"
        assert 'timestamp' in self.logger.fieldnames
        assert os.path.exists(self.test_dir)

    def test_directory_creation(self):
        """Test automatic directory creation."""
        new_dir = os.path.join(self.test_dir, "subdir", "logs")
        logger = CSVLogger(base_directory=new_dir, fieldnames=['test'])
        assert os.path.exists(new_dir)

    def test_filename_generation(self):
        """Test CSV filename generation."""
        filename = self.logger._get_current_filename()
        today = datetime.now().strftime("%Y-%m-%d")
        expected = os.path.join(self.test_dir, f"test_{today}.csv")
        assert filename == expected

    def test_header_creation(self):
        """Test CSV header creation."""
        filename = self.logger._get_current_filename()
        
        # File shouldn't exist yet
        assert not os.path.exists(filename)
        
        # Write header
        self.logger._write_header_if_needed(filename)
        
        # Verify file and header
        assert os.path.exists(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader)
            assert header == self.logger.fieldnames

    def test_log_measurement(self):
        """Test logging measurement data."""
        test_data = {
            'timestamp': 1234567890.0,
            'value': 42.5,
            'status': 'active',
            'extra_field': 'ignored'  # Should be filtered out
        }
        
        self.logger.log_measurement(test_data)
        
        # Verify file creation and content
        filename = self.logger._get_current_filename()
        assert os.path.exists(filename)
        
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            row = next(reader)
            assert float(row['timestamp']) == 1234567890.0
            assert float(row['value']) == 42.5
            assert row['status'] == 'active'
            assert 'extra_field' not in row

    def test_automatic_timestamp(self):
        """Test automatic timestamp generation."""
        test_data = {'value': 100}
        
        with patch('imswitch.imcontrol.controller.loggingutils.datetime') as mock_datetime:
            mock_now = mock_datetime.now.return_value
            mock_now.timestamp.return_value = 9999999999.0
            mock_now.strftime.return_value = "2999-12-31 23:59:59.999"
            
            self.logger.log_measurement(test_data)
        
        # Verify timestamp was added
        filename = self.logger._get_current_filename()
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            row = next(reader)
            assert float(row['timestamp']) == 9999999999.0

    def test_multiple_measurements(self):
        """Test logging multiple measurements."""
        measurements = [
            {'timestamp': 1.0, 'value': 10},
            {'timestamp': 2.0, 'value': 20}, 
            {'timestamp': 3.0, 'value': 30}
        ]
        
        for measurement in measurements:
            self.logger.log_measurement(measurement)
        
        # Verify all measurements
        filename = self.logger._get_current_filename()
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)
            assert len(rows) == 3
            for i, row in enumerate(rows):
                assert float(row['timestamp']) == float(i + 1)
                assert int(row['value']) == (i + 1) * 10

    def test_focus_measurement_logging(self):
        """Test specialized focus measurement logging."""
        self.logger.log_focus_measurement(
            focus_value=123.45,
            timestamp=1234567890.0,
            is_locked=True,
            current_position=50.5,
            pi_output=0.1,
            focus_metric="astigmatism"
        )
        
        filename = self.logger._get_current_filename()
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            row = next(reader)
            assert float(row['timestamp']) == 1234567890.0
            # Note: 'focus_value' is not in default fieldnames, so would be filtered

    def test_thread_safety(self):
        """Test thread-safe logging (basic check)."""
        import threading
        import time
        
        def log_worker(worker_id):
            for i in range(10):
                self.logger.log_measurement({
                    'timestamp': time.time(),
                    'value': worker_id * 100 + i,
                    'status': f'worker_{worker_id}'
                })
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all measurements were logged
        filename = self.logger._get_current_filename()
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)
            assert len(rows) == 30  # 3 workers * 10 measurements each

    def test_file_info_methods(self):
        """Test file information methods."""
        # Initially empty
        assert self.logger.get_record_count() == 0
        assert self.logger.get_file_size() == 0
        
        # Add some data
        self.logger.log_measurement({'timestamp': 1.0, 'value': 100})
        self.logger.log_measurement({'timestamp': 2.0, 'value': 200})
        
        # Check counts
        assert self.logger.get_record_count() == 2
        assert self.logger.get_file_size() > 0

    def test_fieldname_update(self):
        """Test updating fieldnames."""
        new_fieldnames = ['time', 'measurement', 'state']
        self.logger.set_fieldnames(new_fieldnames)
        assert self.logger.fieldnames == new_fieldnames

    def test_no_fieldnames_warning(self):
        """Test logging without fieldnames."""
        logger = CSVLogger(base_directory=self.test_dir, fieldnames=[])
        
        # Should not crash but won't log anything
        logger.log_measurement({'value': 123})
        
        # File shouldn't be created
        filename = logger._get_current_filename()
        assert not os.path.exists(filename)


class TestFocusLockCSVLogger:
    """Test the specialized focus lock CSV logger."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = FocusLockCSVLogger(base_directory=self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_focus_lock_logger_initialization(self):
        """Test focus lock logger initialization."""
        assert self.logger.file_prefix == "focus_lock_measurements"
        assert 'focus_value' in self.logger.fieldnames
        assert 'is_locked' in self.logger.fieldnames
        assert 'current_position' in self.logger.fieldnames
        assert 'pi_output' in self.logger.fieldnames

    def test_log_focus_lock_data(self):
        """Test logging comprehensive focus lock data."""
        self.logger.log_focus_lock_data(
            focus_value=567.89,
            is_locked=True,
            current_position=75.25,
            timestamp=1234567890.0,
            pi_output=0.05,
            error=-2.5,
            step_size_um=0.1
        )
        
        # Verify data was logged
        filename = self.logger._get_current_filename()
        assert os.path.exists(filename)
        
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            row = next(reader)
            assert float(row['focus_value']) == 567.89
            assert row['is_locked'] == 'True'
            assert float(row['current_position']) == 75.25
            assert float(row['pi_output']) == 0.05

    def test_filename_format(self):
        """Test focus lock specific filename format."""
        filename = self.logger._get_current_filename()
        today = datetime.now().strftime("%Y-%m-%d")
        expected = os.path.join(self.test_dir, f"focus_lock_measurements_{today}.csv")
        assert filename == expected

    def test_default_fieldnames_coverage(self):
        """Test that default fieldnames cover expected data."""
        expected_fields = [
            'timestamp', 'datetime', 'focus_value', 'focus_metric',
            'is_locked', 'current_position', 'pi_output', 'error'
        ]
        
        for field in expected_fields:
            assert field in self.logger.fieldnames

    def test_complete_measurement_cycle(self):
        """Test a complete measurement logging cycle."""
        # Log initial measurement (not locked)
        self.logger.log_focus_lock_data(
            focus_value=100.0,
            is_locked=False,
            current_position=50.0
        )
        
        # Log locked measurement
        self.logger.log_focus_lock_data(
            focus_value=105.5,
            is_locked=True, 
            current_position=50.1,
            pi_output=0.02,
            error=1.5,
            step_size_um=0.05
        )
        
        # Verify both measurements
        filename = self.logger._get_current_filename()
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)
            assert len(rows) == 2
            
            # First measurement (not locked)
            assert float(rows[0]['focus_value']) == 100.0
            assert rows[0]['is_locked'] == 'False'
            assert rows[0]['pi_output'] == 'None'
            
            # Second measurement (locked)
            assert float(rows[1]['focus_value']) == 105.5
            assert rows[1]['is_locked'] == 'True'
            assert float(rows[1]['pi_output']) == 0.02