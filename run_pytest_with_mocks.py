#!/usr/bin/env python3
"""
Script to run pytest with all mocks installed.
"""

# Install mocks first
from imswitch import _test_mocks

import subprocess
import sys

# Run pytest with all arguments passed through
if __name__ == "__main__":
    cmd = ["python", "-m", "pytest"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))