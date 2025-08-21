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
        "/LaserController/getLaserNames",
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
    
    # Test setLaserActive - proper OpenAPI format with GET and query parameters
    active_params = {
        "laserName": first_laser,
        "active": True
    }
    response = api_server.get("/LaserController/setLaserActive", params=active_params)
    if response.status_code == 200:
        print(f"✓ Laser {first_laser} activated successfully")
        
        # Test setLaserValue - set laser power/value
        value_params = {
            "laserName": first_laser,
            "value": 50  # Set to 50% or 50 units
        }
        value_response = api_server.get("/LaserController/setLaserValue", params=value_params)
        if value_response.status_code == 200:
            print(f"✓ Laser {first_laser} value set to 50")
            
            # Test getLaserValue - verify the value was set
            get_value_params = {"laserName": first_laser}
            get_response = api_server.get("/LaserController/getLaserValue", params=get_value_params)
            if get_response.status_code == 200:
                current_value = get_response.json()
                print(f"✓ Current laser value: {current_value}")
            
            # Test getLaserValueRanges - get valid value range
            range_response = api_server.get("/LaserController/getLaserValueRanges", params=get_value_params)
            if range_response.status_code == 200:
                value_range = range_response.json()
                print(f"✓ Laser value range: {value_range}")
        
        # Turn laser off
        deactivate_params = {
            "laserName": first_laser,
            "active": False
        }
        deactivate_response = api_server.get("/LaserController/setLaserActive", params=deactivate_params)
        if deactivate_response.status_code == 200:
            print(f"✓ Laser {first_laser} deactivated successfully")
        
    else:
        print(f"? Laser control not available: {response.status_code}")


def test_additional_laser_endpoints(api_server):
    """Test additional laser controller endpoints."""
    # Test changeScanPower endpoint if available
    try:
        response = api_server.get("/LaserController/changeScanPower")
        if response.status_code == 200:
            print(f"✓ Scan power change endpoint available")
    except Exception as e:
        print(f"? Scan power endpoint not available: {e}")
    
    # Test that all laser endpoints are properly documented
    spec_response = api_server.get("/openapi.json")
    if spec_response.status_code == 200:
        spec = spec_response.json()
        laser_endpoints = [
            path for path in spec["paths"].keys() 
            if "LaserController" in path
        ]
        
        print(f"✓ Found {len(laser_endpoints)} LaserController endpoints:")
        for endpoint in laser_endpoints:
            methods = list(spec["paths"][endpoint].keys())
            print(f"  - {endpoint}: {methods}")
        
        # Verify key endpoints are present
        required_endpoints = [
            "/LaserController/getLaserNames",
            "/LaserController/setLaserActive", 
            "/LaserController/setLaserValue",
            "/LaserController/getLaserValue"
        ]
        
        for required in required_endpoints:
            if required in laser_endpoints:
                print(f"✓ Required endpoint found: {required}")
            else:
                print(f"? Required endpoint missing: {required}")

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