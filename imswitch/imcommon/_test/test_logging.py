"""
Unit tests for enhanced logging functionality.
"""

import os
import tempfile
import shutil

from imswitch.imcommon.model import setup_logging, get_log_folder, initLogger


class TestLoggingSetup:
    """Test the logging setup functionality."""

    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.log_folder = os.path.join(self.test_dir, 'logs')

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        # Reset logging handlers
        from imswitch.imcommon.model.logging import baseLogger
        for handler in baseLogger.handlers[:]:
            baseLogger.removeHandler(handler)
            handler.close()

    def test_setup_logging_creates_log_folder(self):
        """Test that setup_logging creates the log folder."""
        setup_logging(
            log_level="INFO",
            log_to_file=True,
            log_folder=self.log_folder,
            config_folder=None
        )

        assert os.path.exists(self.log_folder)
        assert os.path.isdir(self.log_folder)

    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates a log file."""
        setup_logging(
            log_level="INFO",
            log_to_file=True,
            log_folder=self.log_folder,
            config_folder=None
        )

        # Check that at least one log file was created
        log_files = [f for f in os.listdir(self.log_folder) if f.endswith('.log')]
        assert len(log_files) > 0

        # Check filename format
        log_file = log_files[0]
        assert log_file.startswith('imswitch_')
        assert log_file.endswith('.log')

    def test_get_log_folder(self):
        """Test get_log_folder function."""
        log_folder = get_log_folder(config_folder=self.test_dir)
        expected = os.path.join(self.test_dir, 'logs')
        assert log_folder == expected


class TestLoggerAdapter:
    """Test the LoggerAdapter functionality."""

    def test_init_logger_basic(self):
        """Test basic logger initialization."""
        logger = initLogger('test_component')
        assert logger is not None

    def test_init_logger_with_instance_name(self):
        """Test logger initialization with instance name."""
        logger = initLogger('test_component', instanceName='instance1')
        assert logger is not None

    def test_logger_messages(self):
        """Test that logger can log messages at different levels."""
        logger = initLogger('test_messages')

        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
