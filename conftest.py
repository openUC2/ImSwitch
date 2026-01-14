"""
Pytest configuration for ImSwitch tests.

This file provides fixtures and configuration that apply to all tests.
The main purpose is to work around issues with third-party pytest plugins
that try to import modules that aren't installed (like arkitekt_server).
"""
import sys
import pytest


def pytest_configure(config):
    """Configure pytest before test collection."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests requiring server")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "hardware: Tests requiring hardware")


def pytest_collection_modifyitems(config, items):
    """Modify test collection - skip tests that require special conditions."""
    skip_integration = pytest.mark.skip(reason="Integration tests need --run-integration flag")
    skip_hardware = pytest.mark.skip(reason="Hardware tests need --run-hardware flag")
    
    for item in items:
        # Skip hardware tests unless explicitly enabled
        if "hardware" in item.keywords:
            if not config.getoption("--run-hardware", default=False):
                item.add_marker(skip_hardware)
        
        # Integration tests can be enabled with --run-integration
        if "integration" in item.keywords:
            if not config.getoption("--run-integration", default=False):
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require a server"
    )
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require hardware"
    )


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
