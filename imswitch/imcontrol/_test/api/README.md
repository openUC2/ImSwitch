# API Testing for ImSwitch

This directory contains tests for the ImSwitch FastAPI backend, replacing the old Qt-based UI tests.

## Test Structure

- `__init__.py` - Test infrastructure and server management
- `test_general_api.py` - Basic API functionality and system tests  
- `test_detector_api.py` - Detector/camera controller API tests
- `test_positioner_api.py` - Motor/stage positioning API tests

## Running the Tests

### Prerequisites
- ImSwitch installed with requirements
- Test config file (e.g., FRAME2b.json or example_virtual_microscope.json)
- Available ports 8001-8002

### Basic Usage

```bash
# Run all API tests
python3 -m pytest imswitch/imcontrol/_test/api/ -v

# Run specific test file
python3 -m pytest imswitch/imcontrol/_test/api/test_detector_api.py -v

# Run with custom config file
IMSWITCH_CONFIG_FILE="/path/to/FRAME2b.json" python3 -m pytest imswitch/imcontrol/_test/api/ -v
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

## What These Tests Replace

Previously, tests like `test_liveview.py` tested Qt widgets:
```python
qtbot.mouseClick(mainView.widgets['View'].liveviewButton, QtCore.Qt.LeftButton)
```

Now we test the equivalent API endpoints:
```python  
api_server.post("/DetectorController/startLiveview?detectorName=VirtualDetector")
```

## Benefits

- **No Qt dependencies** - Tests run in headless environments (CI/Docker)
- **Real backend testing** - Tests actual FastAPI endpoints used by web frontend
- **Faster execution** - No GUI rendering overhead
- **Better CI integration** - Works in containerized environments
- **API validation** - Ensures REST API contracts are maintained

## Adding New Tests

1. Create test functions that use the `api_server` fixture
2. Make HTTP requests using `api_server.get()`, `api_server.post()`, etc.
3. Assert on response status codes and JSON data
4. Use realistic test data and scenarios

Example:
```python
def test_new_feature(api_server):
    response = api_server.get("/SomeController/getStatus")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
```
