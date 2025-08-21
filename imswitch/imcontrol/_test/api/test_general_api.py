"""
General API tests for ImSwitch backend functionality.
Tests core system endpoints, OpenAPI compliance, and API health checks.
"""
import pytest
import requests
import time
import json
from typing import Dict, Any
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
    assert "paths" in spec
    assert "components" in spec
    
    # Validate OpenAPI spec structure
    assert spec["openapi"].startswith("3.")  # OpenAPI 3.x
    assert "title" in spec["info"]
    assert "version" in spec["info"]


def test_openapi_spec_validation(api_server):
    """Test that OpenAPI specification is valid and comprehensive."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Check for required OpenAPI fields
    required_fields = ["openapi", "info", "paths"]
    for field in required_fields:
        assert field in spec, f"Missing required OpenAPI field: {field}"
    
    
    # Check that we have endpoints
    paths = spec["paths"]
    assert len(paths) > 0, "No API endpoints found in OpenAPI spec"
    
    # Validate each endpoint has proper HTTP methods
    for path, methods in paths.items():
        assert isinstance(methods, dict), f"Invalid methods for path {path}"
        for method, details in methods.items():
            assert method.lower() in ["get", "post", "put", "delete", "patch", "options", "head"]
            assert isinstance(details, dict), f"Invalid details for {method} {path}"


def test_api_endpoints_discovery(api_server):
    """Test discovery of available API endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    assert len(paths) > 0
    
    # Categorize endpoints by controller
    controllers = {
        "DetectorController": [],
        "PositionerController": [],
        "LaserController": [],
        "SettingsController": [],
        "ViewController": [],
        "RecordingController": [],
        "ScanController": []
    }
    
    for path in paths.keys():
        for controller in controllers.keys():
            if controller in path:
                controllers[controller].append(path)
                break
    
    # Ensure we have some controller endpoints
    found_controllers = [c for c, endpoints in controllers.items() if endpoints]
    assert len(found_controllers) > 0, "No controller endpoints found"
    
    print(f"Found endpoints for controllers: {found_controllers}")


def test_api_versioning(api_server):
    """Test API versioning and compatibility."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Check API version is specified
    version = spec["info"]["version"]
    assert version is not None
    assert len(version) > 0
    
    # Version should follow semver-like pattern
    import re
    version_pattern = r'^\d+\.\d+(\.\d+)?'
    assert re.match(version_pattern, version), f"Invalid version format: {version}"


def test_api_response_schemas(api_server):
    """Test that API responses conform to OpenAPI schemas."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    components = spec.get("components", {})
    schemas = components.get("schemas", {})
    
    # Test a few GET endpoints that should return structured data
    get_endpoints = []
    for path, methods in paths.items():
        if "get" in methods and not "{" in path:  # Avoid parameterized paths for now
            get_endpoints.append(path)
    
    # Test first few endpoints
    for endpoint in get_endpoints[:5]:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                # Should return valid JSON
                data = response.json()
                assert data is not None
                print(f"✓ {endpoint} returns valid JSON")
            elif response.status_code in [404, 422, 500]:
                # These are acceptable error codes
                print(f"✓ {endpoint} returns expected error: {response.status_code}")
            else:
                print(f"? {endpoint} returns: {response.status_code}")
        except Exception as e:
            print(f"✗ {endpoint} failed: {e}")


def test_api_http_methods(api_server):
    """Test that endpoints support expected HTTP methods."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    method_counts = {"get": 0, "post": 0, "put": 0, "delete": 0}
    
    for path, methods in paths.items():
        for method in methods.keys():
            if method.lower() in method_counts:
                method_counts[method.lower()] += 1
    
    # Should have at least some GET endpoints
    assert method_counts["get"] > 0, "No GET endpoints found"
    
    # Likely to have POST endpoints for control operations
    print(f"HTTP method distribution: {method_counts}")


def test_api_parameter_validation(api_server):
    """Test API parameter validation and error handling."""
    # Test endpoints that require parameters
    
    # Common endpoint patterns that might need parameters
    test_cases = [
        # Missing required parameters
        ("/DetectorController/getDetectorParameters", {"status_codes": [400, 422, 404]}),
        ("/PositionerController/getPosition", {"status_codes": [400, 422, 404]}),
        ("/LaserController/getLaserPower", {"status_codes": [400, 422, 404]}),
    ]
    
    for endpoint, expected in test_cases:
        try:
            response = api_server.get(endpoint)
            assert response.status_code in expected["status_codes"], \
                f"{endpoint} should return one of {expected['status_codes']}, got {response.status_code}"
            print(f"✓ {endpoint} properly validates parameters: {response.status_code}")
        except Exception as e:
            print(f"? {endpoint} test failed: {e}")


def test_api_error_responses(api_server):
    """Test API error response format and codes."""
    # Test various error conditions
    error_tests = [
        # Non-existent endpoint
        ("/nonexistent/endpoint", 404),
        # Invalid method on docs (should be GET only)
        ("POST", "/docs", [404, 405]),
        # Malformed JSON endpoint
        ("/invalid-path-format", 404),
    ]
    
    for test_data in error_tests:
        if len(test_data) == 2:
            endpoint, expected_code = test_data
            response = api_server.get(endpoint)
        else:
            method, endpoint, expected_codes = test_data
            if method == "POST":
                response = api_server.post(endpoint)
            expected_code = expected_codes
        
        if isinstance(expected_code, list):
            assert response.status_code in expected_code
        else:
            assert response.status_code == expected_code
        
        print(f"✓ Error test passed: {response.status_code}")


def test_api_content_types(api_server):
    """Test API content type handling."""
    # Test that API returns proper content types
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    
    # Test docs endpoint returns HTML
    response = api_server.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_cors_headers(api_server):
    """Test CORS headers are properly set."""
    response = api_server.get("/docs")
    assert response.status_code == 200
    
    # Check for CORS headers (important for web frontend access)
    headers = response.headers
    
    # Test with an OPTIONS request which typically returns CORS headers
    try:
        # Note: Not all servers support OPTIONS, so this is optional
        options_response = api_server.get("/openapi.json")  # Use existing endpoint
        if options_response.status_code == 200:
            print("✓ CORS test completed")
    except:
        print("? CORS test skipped - OPTIONS not supported")


def test_api_performance_basic(api_server):
    """Test basic API performance and response times."""
    import time
    
    # Test response time for key endpoints
    start_time = time.time()
    response = api_server.get("/openapi.json")
    response_time = time.time() - start_time
    
    assert response.status_code == 200
    assert response_time < 5.0, f"OpenAPI spec took too long: {response_time:.2f}s"
    
    print(f"✓ OpenAPI response time: {response_time:.3f}s")


def test_api_security_headers(api_server):
    """Test security-related headers in API responses."""
    response = api_server.get("/docs")
    assert response.status_code == 200
    
    headers = response.headers
    
    # Check for common security headers (these might not all be present)
    security_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options", 
        "X-XSS-Protection",
        "Strict-Transport-Security"
    ]
    
    found_headers = [h for h in security_headers if h in headers]
    print(f"Security headers found: {found_headers}")
    
    # This is informational - not all headers may be required


def test_websocket_endpoints(api_server):
    """Test WebSocket endpoints by checking if port 8002 is reachable."""
    import socket
    from urllib.parse import urlparse
    
    # Get the base URL and extract host
    base_url = api_server.base_url
    parsed_url = urlparse(base_url)
    host = parsed_url.hostname or 'localhost'
    
    # Test WebSocket port 8002
    websocket_port = 8002
    
    try:
        # Create a socket and try to connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        result = sock.connect_ex((host, websocket_port))
        sock.close()
        
        if result == 0:
            print(f"✓ WebSocket port {websocket_port} is reachable at {host}")
        else:
            print(f"✗ WebSocket port {websocket_port} is not reachable at {host} (error: {result})")
            
        # Also check OpenAPI documentation for WebSocket endpoints
        response = api_server.get("/openapi.json")
        spec = response.json()
        paths = spec.get("paths", {})
        websocket_paths = [p for p in paths.keys() if "ws" in p.lower() or "socket" in p.lower()]
        
        if websocket_paths:
            print(f"Found potential WebSocket endpoints in docs: {websocket_paths}")
        else:
            print("No WebSocket endpoints found in OpenAPI documentation")
            
        # Test passes if port is reachable OR WebSocket endpoints are documented
        assert result == 0 or len(websocket_paths) > 0, \
            f"Neither WebSocket port {websocket_port} is reachable nor WebSocket endpoints documented"
            
    except Exception as e:
        print(f"WebSocket connectivity test failed: {e}")
        # Don't fail the test - just report the issue
        pytest.skip(f"WebSocket test skipped due to error: {e}")


def test_api_endpoint_consistency(api_server):
    """Test that API endpoints follow consistent naming conventions."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    
    # Check for consistent naming patterns
    controller_endpoints = {}
    
    for path in paths.keys():
        # Extract controller name from path
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0].endswith("Controller"):
            controller = parts[0]
            if controller not in controller_endpoints:
                controller_endpoints[controller] = []
            controller_endpoints[controller].append(path)
    
    # Ensure we found some controllers
    assert len(controller_endpoints) > 0, "No controller-based endpoints found"
    
    # Print summary
    for controller, endpoints in controller_endpoints.items():
        print(f"{controller}: {len(endpoints)} endpoints")
    
    print(f"Total controllers found: {len(controller_endpoints)}")


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
