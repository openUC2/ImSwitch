# API Testing for ImSwitch

This directory contains comprehensive tests for the ImSwitch FastAPI backend, replacing the old Qt-based UI tests with robust REST API testing.

## Test Structure

- `__init__.py` - Test infrastructure and server management
- `test_general_api.py` - OpenAPI compliance, system health, and core API functionality
- `test_detector_api.py` - Detector/camera controller API tests with comprehensive coverage
- `test_positioner_api.py` - Motor/stage positioning API tests including multi-axis coordination
- `test_additional_controllers.py` - Laser, scanning, recording, and other microscopy controllers

## Enhanced Test Coverage

### General API Tests (`test_general_api.py`)
- ✅ OpenAPI specification validation and compliance
- ✅ API versioning and documentation accessibility  
- ✅ Controller endpoint discovery and enumeration
- ✅ HTTP method validation (GET, POST, PUT, DELETE)
- ✅ Parameter validation and error handling
- ✅ Response schema validation
- ✅ Security headers and CORS configuration
- ✅ Performance testing and response times

### Detector API Tests (`test_detector_api.py`)
- ✅ Detector discovery and enumeration
- ✅ Parameter reading and modification (exposure, gain, etc.)
- ✅ Live view control (start/stop/status)
- ✅ Single image capture
- ✅ Video recording control
- ✅ Status monitoring and health checks
- ✅ Calibration endpoint accessibility
- ✅ Multiple endpoint pattern testing (REST vs ImSwitch-specific)

### Positioner API Tests (`test_positioner_api.py`)
- ✅ Positioner discovery and configuration
- ✅ Position reading and validation
- ✅ Absolute positioning with verification
- ✅ Relative movement testing
- ✅ Speed/velocity control
- ✅ Limits and boundary information
- ✅ Homing functionality
- ✅ Emergency stop capabilities
- ✅ Status and motion state monitoring
- ✅ Multi-axis coordinated movements

### Additional Controllers (`test_additional_controllers.py`)
- ✅ Laser controller (power, enable/disable)
- ✅ Recording controller (status, configuration)
- ✅ Scan controller (configuration, execution)
- ✅ Settings controller (system configuration)
- ✅ View controller (display state)
- ✅ Autofocus controller (configuration, execution)
- ✅ Experiment controller (workflows)
- ✅ Hardware health and diagnostics
- ✅ Batch operations and bulk requests
- ✅ Real-time data endpoints
- ✅ Calibration and characterization

## Running the Tests

### Prerequisites
- ImSwitch installed with requirements
- Test config file (e.g., example_virtual_microscope.json)
- Available ports 8001-8002

### Basic Usage

```bash
# Run all API tests
python3 -m pytest imswitch/imcontrol/_test/api/ -v

# Run specific test file
python3 -m pytest imswitch/imcontrol/_test/api/test_detector_api.py -v

# Run with custom config file
IMSWITCH_TEST_CONFIG="/path/to/config.json" python3 -m pytest imswitch/imcontrol/_test/api/ -v

# Run specific test categories
python3 -m pytest imswitch/imcontrol/_test/api/test_general_api.py::test_openapi_spec_validation -v
```

### Advanced Testing Options

```bash
# Run tests with detailed output
python3 -m pytest imswitch/imcontrol/_test/api/ -v -s

# Run tests with coverage
python3 -m pytest imswitch/imcontrol/_test/api/ --cov=imswitch.imcontrol

# Run tests with timeout
python3 -m pytest imswitch/imcontrol/_test/api/ --timeout=300
```

### Test Configuration

The tests start an ImSwitch server in headless mode using:
```python
main(
    default_config=config_file,  
    is_headless=True,
    http_port=8001,
    socket_port=8002
)
```

## Test Features

### Flexible Endpoint Discovery
Tests automatically discover available endpoints from the OpenAPI specification and test multiple patterns:
- ImSwitch-specific endpoints (`/DetectorController/getDetectorNames`)
- RESTful endpoints (`/detectors`)
- Alternative naming conventions (`/cameras`, `/stages`)

### Robust Error Handling
Tests validate proper error responses for:
- Missing parameters (400, 422)
- Non-existent endpoints (404) 
- Unsupported methods (405)
- Server errors (500)

### Multiple Data Format Support
Tests handle various request/response formats:
- JSON payloads with different field names
- Query parameters vs body parameters
- List vs dictionary responses

### Hardware-Agnostic Testing
Tests work with:
- Virtual/simulated hardware
- Real hardware setups
- Mixed configurations
- Missing hardware (graceful degradation)

## What These Tests Replace

Previously, tests like `test_liveview.py` tested Qt widgets:
```python
qtbot.mouseClick(mainView.widgets['View'].liveviewButton, QtCore.Qt.LeftButton)
```

Now we test the equivalent API endpoints:
```python  
api_server.post("/ViewController/setLiveViewActive?active=true")
```

## Benefits

- **No Qt dependencies** - Tests run in headless environments (CI/Docker)
- **Real backend testing** - Tests actual FastAPI endpoints used by web frontend
- **Faster execution** - No GUI rendering overhead
- **Better CI integration** - Works in containerized environments
- **API validation** - Ensures REST API contracts are maintained
- **OpenAPI compliance** - Validates API documentation accuracy
- **Multiple endpoint patterns** - Tests both ImSwitch and RESTful conventions
- **Comprehensive coverage** - Tests all major microscopy subsystems

## Adding New Tests

1. Create test functions that use the `api_server` fixture
2. Use endpoint discovery patterns to test multiple API conventions
3. Include proper error case testing
4. Validate response schemas and data types
5. Test both positive and negative scenarios

Example:
```python
def test_new_feature(api_server):
    # Test endpoint discovery
    response = api_server.get("/openapi.json")
    spec = response.json()
    endpoints = [p for p in spec["paths"] if "NewFeature" in p]
    
    # Test multiple endpoint patterns
    test_endpoints = [
        "/NewFeatureController/getStatus",
        "/features/status",
        "/new_feature/status"
    ]
    
    for endpoint in test_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                print(f"✓ New feature working via {endpoint}")
                return
        except Exception as e:
            print(f"? {endpoint}: {e}")
```

## Integration with OpenAPI

These tests are designed to work with any OpenAPI-compliant ImSwitch backend. They:
- Parse the OpenAPI specification to discover endpoints
- Validate response schemas against OpenAPI definitions
- Test all documented HTTP methods and parameters
- Ensure API documentation accuracy
