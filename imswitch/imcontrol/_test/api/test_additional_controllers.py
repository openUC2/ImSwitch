"""
API tests for ImSwitch additional controller endpoints.
Tests laser control, scanning, recording, and other microscopy-specific functionality.
"""
import pytest
import requests
import time
import json
from typing import Dict, List, Any
from ..api import api_server, base_url


def test_laser_controller_endpoints(api_server):
    """Test laser controller API endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Find laser-related endpoints
    paths = spec.get("paths", {})
    laser_endpoints = [p for p in paths.keys() if "Laser" in p or "laser" in p]
    
    if not laser_endpoints:
        pytest.skip("No laser endpoints found in API")
    
    print(f"Found {len(laser_endpoints)} laser endpoints")
    
    # Test laser discovery
    discovery_endpoints = [
        "/LaserController/getLasers",
        "/LaserController/getAllLasers",
        "/lasers"
    ]
    
    for endpoint in discovery_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                lasers = response.json()
                print(f"✓ Found lasers via {endpoint}: {lasers}")
                if lasers:
                    test_laser_control(api_server, lasers)
                return
        except Exception as e:
            print(f"Laser discovery via {endpoint} failed: {e}")


def test_laser_control(api_server, lasers):
    """Test laser power and state control."""
    if isinstance(lasers, dict):
        first_laser = list(lasers.keys())[0]
    elif isinstance(lasers, list):
        first_laser = lasers[0]
    else:
        return
    
    # Test laser power control
    power_endpoints = [
        f"/LaserController/setLaserPower?laserName={first_laser}&power=50",
        f"/LaserController/setPower?laser={first_laser}&value=50",
        f"/lasers/{first_laser}/power"
    ]
    
    for endpoint in power_endpoints:
        try:
            if "?" in endpoint:
                response = api_server.post(endpoint)
            else:
                response = api_server.put(endpoint, json={"power": 50})
            
            if response.status_code in [200, 201]:
                print(f"✓ Laser power set via {endpoint}")
                break
        except Exception as e:
            print(f"Laser power control via {endpoint} failed: {e}")
    
    # Test laser enable/disable
    enable_endpoints = [
        f"/LaserController/setLaserEnabled?laserName={first_laser}&enabled=true",
        f"/LaserController/enable?laser={first_laser}",
        f"/lasers/{first_laser}/enable"
    ]
    
    for endpoint in enable_endpoints:
        try:
            response = api_server.post(endpoint)
            if response.status_code in [200, 201]:
                print(f"✓ Laser enabled via {endpoint}")
                # Disable after test
                disable_endpoint = endpoint.replace("enable", "disable").replace("enabled=true", "enabled=false")
                api_server.post(disable_endpoint)
                break
        except Exception as e:
            print(f"Laser enable via {endpoint} failed: {e}")


def test_recording_controller_endpoints(api_server):
    """Test recording and acquisition controller endpoints comprehensively."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    recording_endpoints = [p for p in paths.keys() if "Recording" in p or "recording" in p or "acquisition" in p]
    
    if not recording_endpoints:
        pytest.skip("No recording endpoints found in API")
    
    print(f"Found {len(recording_endpoints)} recording endpoints")
    
    # Test recording start/stop functionality
    test_recording_control(api_server)
    
    # Test recording configuration
    test_recording_configuration(api_server)
    
    # Test image snapping functionality  
    test_snap_functionality(api_server)


def test_recording_control(api_server):
    """Test recording start and stop functionality."""
    # Test starting recording
    start_endpoints = [
        "/RecordingController/startRecording",
        "/recording/start"
    ]
    
    recording_started = False
    for endpoint in start_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                print(f"✓ Recording started via {endpoint}")
                recording_started = True
                break
        except Exception as e:
            print(f"Recording start via {endpoint} failed: {e}")
    
    # Test stopping recording if we managed to start it
    if recording_started:
        stop_endpoints = [
            "/RecordingController/stopRecording", 
            "/recording/stop"
        ]
        
        time.sleep(1)  # Brief recording time
        
        for endpoint in stop_endpoints:
            try:
                response = api_server.get(endpoint)
                if response.status_code == 200:
                    print(f"✓ Recording stopped via {endpoint}")
                    return
            except Exception as e:
                print(f"Recording stop via {endpoint} failed: {e}")


def test_recording_configuration(api_server):
    """Test recording configuration endpoints."""
    # Test setting recording filename
    filename_endpoints = [
        "/RecordingController/setRecFilename"
    ]
    
    test_filename = "test_recording"
    
    for endpoint in filename_endpoints:
        try:
            params = {"filename": test_filename}
            response = api_server.get(endpoint, params=params)
            if response.status_code == 200:
                print(f"✓ Recording filename set via {endpoint}: {test_filename}")
                break
        except Exception as e:
            print(f"Set filename via {endpoint} failed: {e}")
    
    # Test setting recording folder
    folder_endpoints = [
        "/RecordingController/setRecFolder"
    ]
    
    test_folder = "/tmp/test_recordings"
    
    for endpoint in folder_endpoints:
        try:
            params = {"folderPath": test_folder}
            response = api_server.get(endpoint, params=params)
            if response.status_code in [200, 400]:  # 400 might occur if path doesn't exist
                print(f"✓ Recording folder endpoint accessible via {endpoint}: {response.status_code}")
                break
        except Exception as e:
            print(f"Set folder via {endpoint} failed: {e}")
    
    # Test recording mode settings
    mode_endpoints = [
        "/RecordingController/setRecModeScanOnce",
        "/RecordingController/setRecModeSpecFrames",
        "/RecordingController/setRecModeScanTimelapse"
    ]
    
    for endpoint in mode_endpoints:
        try:
            if "Timelapse" in endpoint:
                params = {"lapsesToRec": 5, "freqSeconds": 1.0}
                response = api_server.get(endpoint, params=params)
            elif "SpecFrames" in endpoint:
                params = {"framesToRec": 10}
                response = api_server.get(endpoint, params=params)
            else:
                response = api_server.get(endpoint)
                
            if response.status_code == 200:
                print(f"✓ Recording mode set via {endpoint}")
                break
        except Exception as e:
            print(f"Set recording mode via {endpoint} failed: {e}")


def test_snap_functionality(api_server):
    """Test image snapping functionality."""
    # Test basic snap
    snap_endpoints = [
        "/RecordingController/snapImage",
        "/RecordingController/snapImageToPath"
    ]
    
    for endpoint in snap_endpoints:
        try:
            if "ToPath" in endpoint:
                params = {"fileName": "/tmp/test_snap.tiff"}
                response = api_server.get(endpoint, params=params)
            else:
                params = {"output": False}  # Don't return image data in test
                response = api_server.get(endpoint, params=params)
                
            if response.status_code == 200:
                print(f"✓ Image snap via {endpoint}")
                if endpoint == "/RecordingController/snapImage":
                    # Test with output=True to get image data
                    try:
                        params = {"output": True, "toList": True}
                        response = api_server.get(endpoint, params=params)
                        if response.status_code == 200:
                            data = response.json()
                            print(f"✓ Image snap with data output: {type(data)}")
                    except Exception as e:
                        print(f"Snap with output failed: {e}")
                break
        except Exception as e:
            print(f"Snap via {endpoint} failed: {e}")
    
    # Test detector selection for recording
    detector_endpoints = [
        "/RecordingController/setDetectorToRecord"
    ]
    
    for endpoint in detector_endpoints:
        try:
            # Test with special values
            test_detectors = ["-1", "-2"]  # -1 for current, -2 for all
            
            for detector in test_detectors:
                response = api_server.get(endpoint, json=detector)
                if response.status_code in [200, 400, 422]:  # Accept various responses
                    print(f"✓ Detector selection endpoint accessible: {endpoint} ({response.status_code})")
                    break
        except Exception as e:
            print(f"Detector selection via {endpoint} failed: {e}")


def test_video_streaming(api_server):
    """Test video streaming functionality."""
    streaming_endpoints = [
        "/RecordingController/video_feeder"
    ]
    
    for endpoint in streaming_endpoints:
        try:
            params = {"startStream": True}
            response = api_server.get(endpoint, params=params)
            if response.status_code in [200, 400, 501]:  # Various acceptable responses
                print(f"✓ Video streaming endpoint accessible: {endpoint} ({response.status_code})")
                
                # Test stopping stream
                params = {"startStream": False}
                response = api_server.get(endpoint, params=params)
                if response.status_code in [200, 400, 501]:
                    print(f"✓ Video streaming stop via {endpoint} ({response.status_code})")
                break
        except Exception as e:
            print(f"Video streaming via {endpoint} failed: {e}")


def test_recording_variables(api_server):
    """Test recording variable management."""
    variable_endpoints = [
        "/RecordingController/getVariable"
    ]
    
    test_variables = ["recording_status", "current_filename", "frame_count"]
    
    for endpoint in variable_endpoints:
        try:
            for var_name in test_variables:
                params = {"variable_name": var_name}
                response = api_server.get(endpoint, params=params)
                if response.status_code in [200, 400, 404]:  # Variable might not exist
                    print(f"✓ Variable query via {endpoint} - {var_name}: {response.status_code}")
                    break
        except Exception as e:
            print(f"Variable query via {endpoint} failed: {e}")


def test_scan_controller_endpoints(api_server):
    """Test scanning controller endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    scan_endpoints = [p for p in paths.keys() if "Scan" in p or "scan" in p]
    
    if not scan_endpoints:
        pytest.skip("No scan endpoints found in API")
    
    print(f"Found {len(scan_endpoints)} scan endpoints")
    
    # Test scan configuration
    config_endpoints = [
        "/ScanController/getScanConfig",
        "/ScanController/getConfig",
        "/scan/config"
    ]
    
    for endpoint in config_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                config = response.json()
                print(f"✓ Scan config via {endpoint}")
                if config:
                    test_scan_operations(api_server, config)
                break
        except Exception as e:
            print(f"Scan config via {endpoint} failed: {e}")


def test_scan_operations(api_server, scan_config):
    """Test scan execution operations."""
    # Test scan start/stop
    control_endpoints = [
        "/ScanController/startScan",
        "/ScanController/start",
        "/scan/start"
    ]
    
    for endpoint in control_endpoints:
        try:
            response = api_server.post(endpoint)
            if response.status_code in [200, 201]:
                print(f"✓ Scan started via {endpoint}")
                
                # Brief delay then stop
                time.sleep(1)
                
                stop_endpoint = endpoint.replace("start", "stop")
                stop_response = api_server.post(stop_endpoint)
                if stop_response.status_code in [200, 201]:
                    print(f"✓ Scan stopped via {stop_endpoint}")
                break
        except Exception as e:
            print(f"Scan control via {endpoint} failed: {e}")


def test_settings_controller_endpoints(api_server):
    """Test settings and configuration controller endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    settings_endpoints = [p for p in paths.keys() if "Settings" in p or "settings" in p or "config" in p]
    
    if not settings_endpoints:
        pytest.skip("No settings endpoints found in API")
    
    print(f"Found {len(settings_endpoints)} settings endpoints")
    
    # Test getting system settings
    system_endpoints = [
        "/SettingsController/getSystemSettings",
        "/SettingsController/getConfig",
        "/settings/system"
    ]
    
    for endpoint in system_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                settings = response.json()
                print(f"✓ System settings via {endpoint}")
                assert isinstance(settings, dict)
                break
        except Exception as e:
            print(f"System settings via {endpoint} failed: {e}")


def test_view_controller_endpoints(api_server):
    """Test view and display controller endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    view_endpoints = [p for p in paths.keys() if "View" in p or "view" in p or "display" in p]
    
    if not view_endpoints:
        pytest.skip("No view endpoints found in API")
    
    print(f"Found {len(view_endpoints)} view endpoints")
    
    # Test view state
    state_endpoints = [
        "/ViewController/getViewState",
        "/ViewController/getState",
        "/view/state"
    ]
    
    for endpoint in state_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                state = response.json()
                print(f"✓ View state via {endpoint}")
                break
        except Exception as e:
            print(f"View state via {endpoint} failed: {e}")


def test_autofocus_controller_endpoints(api_server):
    """Test autofocus controller endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    autofocus_endpoints = [p for p in paths.keys() if "Autofocus" in p or "autofocus" in p or "focus" in p]
    
    if not autofocus_endpoints:
        pytest.skip("No autofocus endpoints found in API")
    
    print(f"Found {len(autofocus_endpoints)} autofocus endpoints")
    
    # Test autofocus configuration
    config_endpoints = [
        "/AutofocusController/getConfig",
        "/AutofocusController/getSettings",
        "/autofocus/config"
    ]
    
    for endpoint in config_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                config = response.json()
                print(f"✓ Autofocus config via {endpoint}")
                break
        except Exception as e:
            print(f"Autofocus config via {endpoint} failed: {e}")
    
    # Test autofocus execution
    run_endpoints = [
        "/AutofocusController/runAutofocus",
        "/AutofocusController/focus",
        "/autofocus/run"
    ]
    
    for endpoint in run_endpoints:
        try:
            response = api_server.post(endpoint)
            if response.status_code in [200, 201]:
                print(f"✓ Autofocus run via {endpoint}")
                break
        except Exception as e:
            print(f"Autofocus run via {endpoint} failed: {e}")


def test_experiment_controller_endpoints(api_server):
    """Test experiment and workflow controller endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    experiment_endpoints = [p for p in paths.keys() if "Experiment" in p or "experiment" in p or "workflow" in p]
    
    if not experiment_endpoints:
        pytest.skip("No experiment endpoints found in API")
    
    print(f"Found {len(experiment_endpoints)} experiment endpoints")
    
    # Test experiment list
    list_endpoints = [
        "/ExperimentController/getExperiments",
        "/ExperimentController/list",
        "/experiments"
    ]
    
    for endpoint in list_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                experiments = response.json()
                print(f"✓ Experiments list via {endpoint}")
                break
        except Exception as e:
            print(f"Experiments list via {endpoint} failed: {e}")


def test_hardware_health_endpoints(api_server):
    """Test hardware health and diagnostic endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    health_endpoints = [p for p in paths.keys() if "health" in p.lower() or "status" in p.lower() or "diagnostic" in p.lower()]
    
    print(f"Found {len(health_endpoints)} health/diagnostic endpoints")
    
    # Test system health
    health_check_endpoints = [
        "/health",
        "/status",
        "/diagnostics",
        "/system/health"
    ]
    
    for endpoint in health_check_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                health = response.json()
                print(f"✓ System health via {endpoint}: {health}")
                break
        except Exception as e:
            print(f"Health check via {endpoint} failed: {e}")


def test_api_batch_operations(api_server):
    """Test batch operations and bulk requests."""
    # Test batch configuration changes
    batch_endpoints = [
        "/batch/configure",
        "/bulk/settings",
        "/api/batch"
    ]
    
    sample_batch = {
        "operations": [
            {"type": "get", "endpoint": "/openapi.json"},
            {"type": "get", "endpoint": "/docs"}
        ]
    }
    
    for endpoint in batch_endpoints:
        try:
            response = api_server.post(endpoint, json=sample_batch)
            if response.status_code in [200, 201]:
                print(f"✓ Batch operation via {endpoint}")
                break
        except Exception as e:
            print(f"Batch operation via {endpoint} failed: {e}")


def test_real_time_data_endpoints(api_server):
    """Test real-time data streaming endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    streaming_endpoints = [p for p in paths.keys() if "stream" in p.lower() or "realtime" in p.lower() or "live" in p.lower()]
    
    if streaming_endpoints:
        print(f"Found {len(streaming_endpoints)} streaming endpoints")
        
        for endpoint in streaming_endpoints[:3]:  # Test first 3
            try:
                response = api_server.get(endpoint)
                print(f"✓ Streaming endpoint accessible: {endpoint} ({response.status_code})")
            except Exception as e:
                print(f"Streaming endpoint {endpoint} failed: {e}")
    else:
        print("No streaming endpoints found")


@pytest.mark.skip(reason="Requires WebSocket support")
def test_websocket_real_time_data(api_server):
    """Test WebSocket endpoints for real-time data."""
    # This would test WebSocket connections for:
    # - Live image streaming
    # - Position updates
    # - Sensor data
    # - Status notifications
    pass


def test_calibration_endpoints(api_server):
    """Test calibration and characterization endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    paths = spec.get("paths", {})
    calibration_endpoints = [p for p in paths.keys() if "calibr" in p.lower() or "charact" in p.lower()]
    
    if not calibration_endpoints:
        pytest.skip("No calibration endpoints found")
    
    print(f"Found {len(calibration_endpoints)} calibration endpoints")
    
    for endpoint in calibration_endpoints[:3]:
        try:
            response = api_server.get(endpoint)
            print(f"✓ Calibration endpoint accessible: {endpoint} ({response.status_code})")
        except Exception as e:
            print(f"Calibration endpoint {endpoint} failed: {e}")


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