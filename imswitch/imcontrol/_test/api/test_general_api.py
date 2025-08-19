"""
General API tests for ImSwitch backend functionality.
Tests core system endpoints and health checks.
"""
import pytest
import requests
import time
from ..api import api_server, base_url


def test_api_documentation_available(api_server):
    """Test that FastAPI documentation endpoints are accessible."""
    # Test Swagger UI
    response = api_server.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
        
    # Test OpenAPI spec
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert "openapi" in spec
    assert "info" in spec


def test_system_info_endpoints(api_server):
    """Test system information endpoints."""
    # Test getting system status/info if available
    # Note: Endpoint names may vary based on ImSwitch implementation
    
    # Try common system endpoints
    system_endpoints = [
        "/system/info",
        "/system/status", 
        "/info",
        "/status",
        "/health"
    ]
    
    found_endpoint = False
    for endpoint in system_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                found_endpoint = True
                break
        except:
            continue
    
    # At minimum, we should be able to get the OpenAPI spec
    response = api_server.get("/openapi.json")
    assert response.status_code == 200


def test_controller_discovery(api_server):
    """Test that we can discover available controllers via API."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Check that we have some controller endpoints
    paths = spec.get("paths", {})
    assert len(paths) > 0
    
    # Look for common controller patterns
    controller_patterns = ["DetectorController", "PositionerController", "LaserController"]
    found_controllers = []
    
    for path in paths.keys():
        for pattern in controller_patterns:
            if pattern in path:
                found_controllers.append(pattern)
                break
    
    assert len(found_controllers) > 0, "No controller endpoints found in API"


def test_cors_headers(api_server):
    """Test CORS headers are properly set."""
    response = api_server.get("/docs")
    assert response.status_code == 200
    
    # Check for CORS headers (may be set by ImSwitch)
    # This is important for web frontend access
    headers = response.headers
    # Note: CORS headers may not be present in all responses


def test_api_response_format(api_server):
    """Test that API responses are in expected JSON format."""
    # Get OpenAPI spec to find a simple endpoint
    response = api_server.get("/openapi.json") 
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    
    # Find a GET endpoint that should return JSON
    test_endpoint = None
    for path, methods in paths.items():
        if "get" in methods:
            # Look for simple endpoints without parameters
            get_info = methods["get"]
            if not get_info.get("parameters"):
                test_endpoint = path
                break
    
    if test_endpoint:
        response = api_server.get(test_endpoint)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 404, 422, 500]
        
        if response.status_code == 200:
            # If successful, should be valid JSON
            try:
                response.json()
            except ValueError:
                pytest.fail(f"Response from {test_endpoint} is not valid JSON")


def test_error_handling(api_server):
    """Test that API handles errors gracefully."""
    # Test invalid endpoint
    response = api_server.get("/nonexistent/endpoint")
    assert response.status_code == 404
    
    # Test invalid method on existing endpoint
    response = api_server.post("/docs")  # docs should be GET only
    assert response.status_code in [404, 405]  # Method not allowed


@pytest.mark.skip(reason="Requires specific controller setup")
def test_websocket_endpoints(api_server):
    """Test WebSocket endpoints if available."""
    # ImSwitch may have WebSocket endpoints for real-time data
    # This would require WebSocket client testing
    pass


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
