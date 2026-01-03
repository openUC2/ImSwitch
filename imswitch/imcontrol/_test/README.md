# ImSwitch Test Suite

## Test Architecture

The ImSwitch test suite is organized into two main categories:

### Unit Tests (`imswitch/imcontrol/_test/unit/`)

Self-contained tests that don't require a running ImSwitch server:

| Test File | Description | Status |
|-----------|-------------|--------|
| `pixelcalibration/test_apriltag_grid.py` | AprilTag grid calibration system | ✅ Active |
| `pixelcalibration/test_overview_calibrator.py` | Overview calibrator | ✅ Active |
| `test_acquisition.py` | Detector acquisition tests | ✅ Active |
| `test_loggingutils.py` | CSV logging utilities | ✅ Active |
| `test_pidcontroller.py` | PID controller logic | ✅ Active |
| `test_storage_manager.py` | Storage management | ✅ Active |
| `test_democontroller.py_` | Demo controller (disabled) | ❌ Needs refactoring |
| `test_lepmon_hmi.py_` | Lepmon HMI (disabled) | ❌ Needs refactoring |
| `test_recording.py_` | Recording tests (disabled) | ❌ Needs Qt/signals |
| `test_scan.py_` | Scan tests (disabled) | ❌ Broken |
| `test_stores.py_` | Store tests (disabled) | ⚠️ Needs review |

### API/Integration Tests (`imswitch/imcontrol/_test/api/`)

Integration tests that require a running ImSwitch server (headless mode):

| Test File | Description |
|-----------|-------------|
| `test_general_api.py` | OpenAPI spec, documentation endpoints |
| `test_detector_api.py` | Detector controller endpoints |
| `test_positioner_api.py` | Positioner controller endpoints |
| `test_storage_api.py` | Storage controller endpoints |
| `test_experimentcontroller.py` | Experiment controller |
| `test_additional_controllers.py` | Other controller endpoints |

**Note:** API tests are not run in CI by default due to the complexity of starting
a full ImSwitch server. They can be run locally with:

```bash
python3 -m pytest --pyargs imswitch.imcontrol._test.api -v
```

## Running Tests

### Prerequisites

```bash
pip install pytest ruff
```

### Run Unit Tests (Recommended)

```bash
# From project root directory
python3 -m pytest --pyargs imswitch.imcontrol._test.unit -v

# Or with explicit plugin disable (if arkitekt-next is installed)
python3 -m pytest --pyargs imswitch.imcontrol._test.unit -p no:arkitekt_next -v
```

### Run Specific Test File

```bash
python3 -m pytest --pyargs imswitch.imcontrol._test.unit.test_pidcontroller -v
```

### Run Tests with Coverage

```bash
pip install pytest-cov
python3 -m pytest --pyargs imswitch.imcontrol._test.unit --cov=imswitch -v
```

## Known Issues

### arkitekt_next pytest plugin

The `arkitekt-next` package registers a pytest plugin that tries to import 
`arkitekt_server`, which may not be installed. This causes pytest to fail
at startup with `ModuleNotFoundError: No module named 'arkitekt_server'`.

**Solution:** Disable the plugin with `-p no:arkitekt_next` or configure it
in `pytest.ini` (already done in this repository).

### Disabled Tests (`.py_` suffix)

Some tests are disabled by renaming them with a `.py_` suffix because they:

1. Use hacky import mechanisms that don't work with relative imports
2. Depend on Qt signals that require a running application
3. Have broken assertions or logic

These tests need to be refactored to work properly.

## CI Configuration

The GitHub Actions workflow (`.github/workflows/imswitch-test.yml`) runs:

1. **Linting with ruff** - Checks for syntax errors and undefined names
2. **Unit tests** - Runs all tests in `imswitch.imcontrol._test.unit`

Tests run on:
- macOS (Intel and Apple Silicon)
- Windows
- Python 3.11 and 3.12

## Adding New Tests

1. Place unit tests in `imswitch/imcontrol/_test/unit/`
2. Use the `test_*.py` naming convention
3. Import modules using proper Python import paths (not file paths)
4. Use pytest fixtures from `conftest.py` when applicable
5. Add appropriate markers for slow or hardware-dependent tests:

```python
import pytest

@pytest.mark.slow
def test_long_running_operation():
    ...

@pytest.mark.hardware
def test_requires_hardware():
    ...
```

## Copyright

Copyright (C) 2020-2024 ImSwitch developers

This file is part of ImSwitch and is licensed under the GNU General Public License v3.
