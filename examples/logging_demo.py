#!/usr/bin/env python
"""
Example script demonstrating the ImSwitch logging system.

This script shows how to use the enhanced logging features including:
- Configurable log levels
- File logging
- Signal emission (for headless mode)
"""

import sys
import os
import tempfile

# Add ImSwitch to path (adjust this path as needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imswitch.imcommon.model.logging import setup_logging, initLogger, get_log_folder


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("=" * 60)
    print("Demo 1: Basic Logging")
    print("=" * 60)

    # Setup logging with INFO level
    test_dir = tempfile.mkdtemp()
    log_folder = get_log_folder(config_folder=test_dir)

    setup_logging(
        log_level="INFO",
        log_to_file=True,
        log_folder=log_folder,
        config_folder=None
    )

    # Create a logger
    logger = initLogger('DemoComponent')

    # Log messages at different levels
    logger.debug("This is a DEBUG message (won't show with INFO level)")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    print(f"\nLog file created at: {log_folder}")
    print(f"Log files: {os.listdir(log_folder)}")

    return test_dir


def main():
    """Run all logging demos."""
    print("\nImSwitch Logging System Demo")
    print("=" * 60)

    temp_dirs = []

    try:
        # Run demo
        temp_dirs.append(demo_basic_logging())

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

        print("\nNote: In production, log files are stored in:")
        print("  ~/ImSwitchConfig/logs/")
        print("\nYou can access log files via REST API:")
        print("  GET /LogController/listLogFiles")
        print("  GET /LogController/downloadLogFile?filename=<filename>")

    finally:
        # Cleanup temp directories
        import shutil
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        print("\nTemporary directories cleaned up.")


if __name__ == "__main__":
    main()
