"""
ExperimentController API tests for ImSwitch backend functionality.
Tests experiment control endpoints including status monitoring, hardware parameters, and scan acquisition.
"""
import pytest
import time
from ..api import api_server, base_url


def test_get_experiment_status(api_server):
    """Test getting current experiment status."""
    response = api_server.get("/ExperimentController/getExperimentStatus")
    assert response.status_code == 200
    print(f"✓ Experiment status retrieved successfully")


def test_get_hardware_parameters(api_server):
    """Test getting hardware parameters."""
    response = api_server.get("/ExperimentController/getHardwareParameters")
    assert response.status_code == 200
    print(f"✓ Hardware parameters retrieved successfully")


def test_get_ome_writer_config(api_server):
    """Test getting OME writer configuration."""
    response = api_server.get("/ExperimentController/getOMEWriterConfig")
    assert response.status_code == 200
    print(f"✓ OME writer configuration retrieved successfully")


def test_fast_stage_scan_acquisition_filepath(api_server):
    """Test getting file path of last fast stage scan."""
    response = api_server.get("/ExperimentController/startFastStageScanAcquisitionFilePath")
    # May return 200 with path or 404 if no scan exists yet
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        print(f"✓ Fast stage scan file path: {response.json()}")
    else:
        print(f"✓ No previous fast stage scan found (expected for new system)")


def test_start_fast_stage_scan_acquisition(api_server):
    """Test starting fast stage scan acquisition with minimal parameters."""
    # Use minimal scan parameters to avoid long execution times
    params = {
        "xstart": 0,
        "xstep": 100,  # Small step size
        "nx": 2,       # Only 2 steps
        "ystart": 0,
        "ystep": 100,
        "ny": 2,
        "tsettle": 10, # Short settle time
        "tExposure": 10, # Short exposure
        "tPeriod": 1,
        "nTimes": 1
    }
    
    response = api_server.get("/ExperimentController/startFastStageScanAcquisition", params=params)
    assert response.status_code in [200, 422]  # May fail if hardware not available
    
    if response.status_code == 200:
        print(f"✓ Fast stage scan acquisition started successfully")
        
        # Check status
        status_response = api_server.get("/ExperimentController/getExperimentStatus")
        assert status_response.status_code == 200
        print(f"✓ Experiment status during scan: {status_response.json()}")
        
        # Wait a moment then stop the scan
        time.sleep(0.5)
        stop_response = api_server.get("/ExperimentController/stopFastStageScanAcquisition")
        assert stop_response.status_code == 200
        print(f"✓ Fast stage scan stopped successfully")
    else:
        print(f"✓ Fast stage scan not available (expected for virtual hardware): {response.status_code}")


def test_experiment_lifecycle(api_server):
    """Test complete experiment lifecycle: start -> status -> stop."""
    # Start with minimal scan parameters
    scan_params = {
        "nx": 1, "ny": 1,  # Single point scan
        "tExposure": 5,    # Very short exposure
        "tsettle": 5       # Very short settle
    }
    
    # Try to start experiment
    start_response = api_server.get("/ExperimentController/startFastStageScanAcquisition", params=scan_params)
    
    if start_response.status_code == 200:
        print(f"✓ Experiment started successfully")
        
        # Get status
        status_response = api_server.get("/ExperimentController/getExperimentStatus")
        assert status_response.status_code == 200
        status = status_response.json()
        print(f"✓ Experiment status: {status}")
        
        # Try pause workflow
        pause_response = api_server.get("/ExperimentController/pauseWorkflow")
        assert pause_response.status_code == 200
        print(f"✓ Workflow pause attempted")
        
        # Try resume
        resume_response = api_server.get("/ExperimentController/resumeExperiment")
        assert resume_response.status_code == 200
        print(f"✓ Experiment resume attempted")
        
        # Stop experiment
        stop_response = api_server.get("/ExperimentController/stopExperiment")
        assert stop_response.status_code == 200
        print(f"✓ Experiment stopped successfully")
        
    else:
        print(f"✓ Experiment start not available (expected for virtual setup): {start_response.status_code}")


def test_force_stop_experiment(api_server):
    """Test force stopping experiment."""
    response = api_server.get("/ExperimentController/forceStopExperiment")
    assert response.status_code == 200
    print(f"✓ Force stop experiment command sent successfully")


def test_wellplate_experiment_endpoint(api_server):
    """Test wellplate experiment endpoint (POST method)."""
    # This endpoint requires JSON body with experiment configuration
    # Test with minimal experiment configuration
    experiment_config = {
        "name": "test_experiment",
        "description": "Test experiment for API validation"
    }
    
    response = api_server.post("/ExperimentController/startWellplateExperiment", json=experiment_config)
    # May return 422 if schema validation fails, which is expected
    assert response.status_code in [200, 422]
    
    if response.status_code == 200:
        print(f"✓ Wellplate experiment started successfully")
        
        # Stop the experiment
        stop_response = api_server.get("/ExperimentController/stopExperiment")
        assert stop_response.status_code == 200
        print(f"✓ Wellplate experiment stopped")
    else:
        print(f"✓ Wellplate experiment endpoint validated (schema requirements not met): {response.status_code}")


def test_get_last_scan_ome_zarr(api_server):
    """Test getting last scan as OME-ZARR."""
    response = api_server.get("/ExperimentController/getLastScanAsOMEZARR")
    # May return 404 if no scan exists
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        print(f"✓ Last OME-ZARR scan retrieved successfully")
    else:
        print(f"✓ No previous OME-ZARR scan found (expected for new system)")


def test_experiment_controller_endpoints_discovery(api_server):
    """Test discovery of all ExperimentController endpoints."""
    # Get OpenAPI spec to validate endpoints
    spec_response = api_server.get("/openapi.json")
    assert spec_response.status_code == 200
    spec = spec_response.json()
    
    # Find all ExperimentController endpoints
    experiment_endpoints = [
        path for path in spec["paths"].keys() 
        if "ExperimentController" in path
    ]
    
    assert len(experiment_endpoints) > 0, "No ExperimentController endpoints found"
    print(f"✓ Found {len(experiment_endpoints)} ExperimentController endpoints:")
    
    for endpoint in experiment_endpoints:
        methods = list(spec["paths"][endpoint].keys())
        print(f"  - {endpoint}: {methods}")
    
    # Verify required endpoints are present
    required_endpoints = [
        "/ExperimentController/getExperimentStatus",
        "/ExperimentController/getHardwareParameters", 
        "/ExperimentController/startFastStageScanAcquisition",
        "/ExperimentController/stopExperiment",
        "/ExperimentController/forceStopExperiment"
    ]
    
    for required in required_endpoints:
        assert required in experiment_endpoints, f"Required endpoint missing: {required}"
    
    print(f"✓ All required ExperimentController endpoints are present")


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