"""
API testing utilities for headless ImSwitch FastAPI backend.
Replaces Qt-based UI tests with HTTP endpoint testing.
"""
import asyncio
import threading
import time
from typing import Optional, Dict, Any
import requests
import pytest
from imswitch.__main__ import main
from pathlib import Path
import os 

class ImSwitchAPITestServer:
    """Test server that starts ImSwitch in headless mode for API testing."""
    
    def __init__(self, config_file: str = None, 
                 http_port: int = 8001, socket_port: int = 8002):
        
        # have configfile from ./_data/user_defaults/imcontrol_setups/example_virtual_microscope.json
        # Automatically find config file if not provided
        if config_file is None or config_file == "example_virtual_microscope.json":
            # Use automatic path resolution for default config
            self.config_file = self.get_default_config_path()
        elif not os.path.isabs(config_file) and not os.path.exists(config_file):
            # If it's just a filename, try to find it in the default location
            self.config_file = self.get_config_path_by_name(config_file)
        else:
            # Use provided path as-is
            self.config_file = config_file
            
        print(f"Using config file: {self.config_file}")
        self.http_port = http_port
        self.socket_port = socket_port
        self.server_thread: Optional[threading.Thread] = None
        self.base_url = f"http://localhost:{http_port}"
        self.is_running = False
        
    def start(self, timeout: int = 60):
        """Start ImSwitch server in background thread."""
        if self.is_running:
            return
            
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/docs", timeout=2)
                if response.status_code == 200:
                    self.is_running = True
                    print(f"ImSwitch API server ready at {self.base_url}")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            
        raise TimeoutError(f"ImSwitch server failed to start within {timeout}s")
        
    def get_default_config_path(self):
        """Get config file path from environment variable or use default."""
        # Check environment variable first
        env_config = os.environ.get('IMSWITCH_TEST_CONFIG')
        if env_config and Path(env_config).exists():
            return env_config
        
        # Try to find the default config file automatically
        possible_paths = [
            # Relative to current working directory
            Path.cwd() / '_data' / 'user_defaults' / 'imcontrol_setups' / 'example_virtual_microscope.json',
            # Relative to this file's location
            Path(__file__).parents[3] / '_data' / 'user_defaults' / 'imcontrol_setups' / 'example_virtual_microscope.json',
            # Common installation locations
            Path.home() / 'ImSwitchConfig' / 'imcontrol_setups' / 'example_virtual_microscope.json',
            Path('/tmp/ImSwitchConfig/imcontrol_setups/example_virtual_microscope.json'),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # If no file found, return the most likely default path
        default_path = Path(__file__).parents[3] / '_data' / 'user_defaults' / 'imcontrol_setups' / 'example_virtual_microscope.json'
        print(f"Warning: Config file not found. Using default path: {default_path}")
        return str(default_path)
    
    def get_config_path_by_name(self, filename: str):
        """Find config file by name in standard locations."""
        # Try to find the config file by name in standard locations
        possible_dirs = [
            # Relative to current working directory
            Path.cwd() / '_data' / 'user_defaults' / 'imcontrol_setups',
            # Relative to this file's location  
            Path(__file__).parents[3] / '_data' / 'user_defaults' / 'imcontrol_setups',
            # Common installation locations
            Path.home() / 'ImSwitchConfig' / 'imcontrol_setups',
            Path('/tmp/ImSwitchConfig/imcontrol_setups'),
        ]
        
        for config_dir in possible_dirs:
            config_path = config_dir / filename
            if config_path.exists():
                return str(config_path)
                
        # If not found, return path in most likely location
        default_path = Path(__file__).parents[3] / '_data' / 'user_defaults' / 'imcontrol_setups' / filename
        print(f"Warning: Config file '{filename}' not found. Using default path: {default_path}")
        return str(default_path)

    def _run_server(self):
        """Run ImSwitch main function in headless mode."""
        try:
            main(
                default_config=self.config_file,
                is_headless=True,
                http_port=self.http_port,
                socket_port=self.socket_port, 
                ssl=False,  # Disable SSL for testing
            )
        except Exception as e:
            print(f"Server startup error: {e}")
            
    def stop(self):
        """Stop the server (note: may require process termination)."""
        self.is_running = False
        # Note: ImSwitch doesn't have clean shutdown, may need process kill
        
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make GET request to API endpoint."""
        return requests.get(f"{self.base_url}{endpoint}", **kwargs)
        
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make POST request to API endpoint."""
        return requests.post(f"{self.base_url}{endpoint}", **kwargs)
        
    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """Make PUT request to API endpoint.""" 
        return requests.put(f"{self.base_url}{endpoint}", **kwargs)


# Global server instance for tests
_test_server: Optional[ImSwitchAPITestServer] = None


def get_test_server(config_file: str = None) -> ImSwitchAPITestServer:
    """Get or create test server instance."""
    global _test_server
    if _test_server is None:
        _test_server = ImSwitchAPITestServer(config_file=config_file)
    return _test_server


@pytest.fixture(scope="session")
def api_server():
    """Pytest fixture that provides running ImSwitch API server."""
    server = get_test_server()
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="session") 
def base_url(api_server):
    """Pytest fixture that provides base URL for API requests."""
    return api_server.base_url



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
