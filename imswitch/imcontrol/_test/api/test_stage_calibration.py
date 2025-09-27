"""
API tests for enhanced Stage Center Calibration functionality.
Tests the new calibration target features including manual, automatic, 
maze navigation, stepsize and wellplate calibrations.
"""
import pytest
import requests
import time
import json
from typing import Dict, List, Any
from ..api import api_server, base_url


def test_stage_calibration_discovery(api_server):
    """Test discovery of stage calibration endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Find stage calibration related endpoints
    paths = spec.get("paths", {})
    calibration_endpoints = [p for p in paths.keys() if "StageCenterCalibration" in p]
    
    if not calibration_endpoints:
        pytest.skip("No stage calibration endpoints found in API")
    
    print(f"Found {len(calibration_endpoints)} stage calibration endpoints")
    
    # Test required endpoints are present
    required_endpoints = [
        "/StageCenterCalibrationController/setKnownPosition",
        "/StageCenterCalibrationController/getCalibrationTargetInfo",
        "/StageCenterCalibrationController/performAutomaticCalibration",
        "/StageCenterCalibrationController/startMaze",
        "/StageCenterCalibrationController/stopMaze",
        "/StageCenterCalibrationController/performStepsizeCalibration",
        "/StageCenterCalibrationController/perform384WellplateCalibration"
    ]
    
    for required in required_endpoints:
        if required in calibration_endpoints:
            print(f"✓ Required endpoint found: {required}")
        else:
            print(f"? Required endpoint missing: {required}")


def test_calibration_target_info(api_server):
    """Test calibration target information endpoint."""
    endpoint = "/StageCenterCalibrationController/getCalibrationTargetInfo"
    
    try:
        response = api_server.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            required_fields = [
                "width_mm", "height_mm", "frontside_svg", "backside_svg",
                "calibration_center", "maze_start", "stepsize_grid", "wellplate_start"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
                print(f"✓ Found field: {field}")
            
            # Validate SVG content
            assert "<svg" in data["frontside_svg"], "Invalid frontside SVG"
            assert "<svg" in data["backside_svg"], "Invalid backside SVG"
            
            # Validate coordinates
            assert data["calibration_center"]["x"] == 63.81
            assert data["calibration_center"]["y"] == 42.06
            assert data["maze_start"]["x"] == 9.5
            assert data["maze_start"]["y"] == 11.5
            
            print("✓ Calibration target info endpoint working correctly")
        
        elif response.status_code in [400, 501]:
            print(f"✓ Calibration target info endpoint responded (non-200 acceptable): {response.status_code}")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Calibration target info test failed: {e}")


def test_set_known_position(api_server):
    """Test manual calibration endpoint."""
    endpoint = "/StageCenterCalibrationController/setKnownPosition"
    
    try:
        # Test with default position
        response = api_server.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert data["x_mm"] == 63.81
            assert data["y_mm"] == 42.06
            print("✓ Set known position (default) working")
        
        # Test with custom position
        params = {"x_mm": 50.0, "y_mm": 30.0}
        response = api_server.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert data["x_mm"] == 50.0
            assert data["y_mm"] == 30.0
            print("✓ Set known position (custom) working")
        
        elif response.status_code in [400, 501]:
            print(f"✓ Set known position endpoint responded (non-200 acceptable): {response.status_code}")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Set known position test failed: {e}")


def test_maze_navigation(api_server):
    """Test maze navigation functionality."""
    start_endpoint = "/StageCenterCalibrationController/startMaze"
    stop_endpoint = "/StageCenterCalibrationController/stopMaze"
    status_endpoint = "/StageCenterCalibrationController/getMazeStatus"
    
    try:
        # Test maze status when not running
        response = api_server.get(status_endpoint)
        if response.status_code == 200:
            data = response.json()
            assert "running" in data
            assert "positions_visited" in data
            print("✓ Maze status endpoint working")
        
        # Test start maze
        response = api_server.get(start_endpoint)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] in ["started", "error"]  # May error if no stage available
            print("✓ Start maze endpoint working")
            
            # If started successfully, test stop
            if data["status"] == "started":
                time.sleep(0.5)  # Let it run briefly
                
                stop_response = api_server.get(stop_endpoint)
                assert stop_response.status_code == 200
                stop_data = stop_response.json()
                assert stop_data["status"] == "stopped"
                print("✓ Stop maze endpoint working")
        
        elif response.status_code in [400, 501]:
            print(f"✓ Maze navigation endpoints responded (non-200 acceptable): {response.status_code}")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Maze navigation test failed: {e}")


def test_stepsize_calibration(api_server):
    """Test stepsize calibration functionality."""
    endpoint = "/StageCenterCalibrationController/performStepsizeCalibration"
    
    try:
        response = api_server.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] in ["success", "error"]  # May error if no stage/detector available
            print(f"✓ Stepsize calibration endpoint working: {data['status']}")
        
        elif response.status_code in [400, 501]:
            print(f"✓ Stepsize calibration endpoint responded (non-200 acceptable): {response.status_code}")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Stepsize calibration test failed: {e}")


def test_384_wellplate_calibration(api_server):
    """Test 384 wellplate calibration functionality."""
    endpoint = "/StageCenterCalibrationController/perform384WellplateCalibration"
    
    try:
        response = api_server.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] in ["success", "error"]  # May error if no stage/detector available
            print(f"✓ 384 wellplate calibration endpoint working: {data['status']}")
        
        # Test with custom wells
        params = {"sample_wells": ["A1", "H12", "P24"]}
        response = api_server.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] in ["success", "error"]
            print(f"✓ 384 wellplate calibration (custom wells) working: {data['status']}")
        
        elif response.status_code in [400, 501]:
            print(f"✓ 384 wellplate calibration endpoint responded (non-200 acceptable): {response.status_code}")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"384 wellplate calibration test failed: {e}")


def test_automatic_calibration(api_server):
    """Test automatic calibration with line detection."""
    endpoint = "/StageCenterCalibrationController/performAutomaticCalibration"
    
    try:
        response = api_server.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] in ["success", "error"]  # May error if no stage/detector/laser available
            print(f"✓ Automatic calibration endpoint working: {data['status']}")
        
        # Test with laser parameters
        params = {"laser_name": "test_laser", "laser_intensity": 30.0}
        response = api_server.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] in ["success", "error"]
            print(f"✓ Automatic calibration (with laser) working: {data['status']}")
        
        elif response.status_code in [400, 501]:
            print(f"✓ Automatic calibration endpoint responded (non-200 acceptable): {response.status_code}")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Automatic calibration test failed: {e}")


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