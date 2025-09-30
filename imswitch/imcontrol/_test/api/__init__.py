"""
API testing utilities for headless ImSwitch FastAPI backend.
Replaces Qt-based UI tests with HTTP endpoint testing.
"""
import asyncio
import threading
import time
import logging
import sys
from typing import Optional, Dict, Any
import requests
import pytest
from imswitch.__main__ import main
from pathlib import Path
import os 

class ImSwitchAPITestServer:
    """Test server that starts ImSwitch in headless mode for API testing."""
    
    def __init__(self, config_file: str = None, 
                 http_port: int = 8001, socket_port: int = 8002, ssl: bool = False):

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
        self.ssl = ssl
        self.server_thread: Optional[threading.Thread] = None
        self.base_url = f"http://localhost:{http_port}"
        self.is_running = False
        
    def start(self, timeout: int = 160):
        """Start ImSwitch server in background thread."""
        if self.is_running:
            print("[TEST SERVER] Server already running")
            return
            
        # Check if ports are available before starting
        import socket
        for port_name, port in [("HTTP", self.http_port), ("Socket", self.socket_port)]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                except OSError as e:
                    raise RuntimeError(f"{port_name} port {port} is already in use. "
                                     f"Another ImSwitch instance may be running. Error: {e}")
            
        print(f"[TEST SERVER] Starting ImSwitch API test server...")
        print(f"[TEST SERVER] Config file: {self.config_file}")
        print(f"[TEST SERVER] HTTP port: {self.http_port}")
        print(f"[TEST SERVER] Socket port: {self.socket_port}")
        print(f"[TEST SERVER] SSL: {self.ssl}")

        # Start server in background thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="ImSwitchTestServer"
        )
        self.server_thread.start()
        print(f"[TEST SERVER] Server thread started, waiting for server to be ready...")
        
        # Wait for server to be ready
        start_time = time.time()
        last_status_time = start_time
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/docs", timeout=2)
                if response.status_code == 200:
                    self.is_running = True
                    elapsed = time.time() - start_time
                    print(f"[TEST SERVER] ImSwitch API server ready at {self.base_url} (took {elapsed:.1f}s)")
                    return
            except requests.exceptions.RequestException as e:
                # Show periodic status updates
                current_time = time.time()
                if current_time - last_status_time > 10:  # Every 10 seconds
                    elapsed = current_time - start_time
                    print(f"[TEST SERVER] Still waiting for server... ({elapsed:.1f}s elapsed, last error: {type(e).__name__})")
                    last_status_time = current_time
                    
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
        # Configure logging to ensure thread output is visible
        thread_logger = logging.getLogger('imswitch_test_server')
        thread_logger.setLevel(logging.DEBUG)
        thread_logger.info(f"Starting ImSwitch server in thread...")
        thread_logger.info(f"Config file: {self.config_file}")
        thread_logger.info(f"HTTP port: {self.http_port}")
        thread_logger.info(f"Socket port: {self.socket_port}")
        
        try:
            print(f"[TEST SERVER] Starting ImSwitch server in headless mode...", flush=True)
            print(f"[TEST SERVER] Config: {self.config_file}", flush=True)
            print(f"[TEST SERVER] HTTP Port: {self.http_port}", flush=True)
            
            main(
                default_config=self.config_file,
                is_headless=True,
                http_port=self.http_port,
                socket_port=self.socket_port, 
                ssl=self.ssl,  # Fixed: was self.is_ssl
            )
        except Exception as e:
            error_msg = f"Server startup error: {e}"
            thread_logger.error(error_msg)
            print(f"[TEST SERVER ERROR] {error_msg}", flush=True)
            
            # Print traceback for debugging
            import traceback
            tb = traceback.format_exc()
            thread_logger.error(f"Full traceback:\n{tb}")
            print(f"[TEST SERVER TRACEBACK]\n{tb}", flush=True)
            raise  # Re-raise to ensure the error is visible
            
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
        # Use dynamic port allocation to avoid conflicts
        import socket
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        http_port = find_free_port()
        socket_port = find_free_port()
        
        print(f"[TEST SERVER] Creating new test server instance on ports {http_port}/{socket_port}")
        _test_server = ImSwitchAPITestServer(
            config_file=config_file,
            http_port=http_port,
            socket_port=socket_port
        )
    return _test_server


@pytest.fixture(scope="session")
def api_server():
    """Pytest fixture that provides running ImSwitch API server.
    
    Uses session scope to ensure only one server instance per pytest session.
    The server runs in a daemon thread and will be cleaned up when pytest exits.
    """
    print(f"[PYTEST FIXTURE] Initializing session-scoped API server...")
    server = get_test_server()
    
    try:
        server.start()
        print(f"[PYTEST FIXTURE] API server started successfully at {server.base_url}")
        yield server
    finally:
        print(f"[PYTEST FIXTURE] Cleaning up API server...")
        server.stop()
        # Note: ImSwitch doesn't have graceful shutdown, daemon thread will be terminated by pytest


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
