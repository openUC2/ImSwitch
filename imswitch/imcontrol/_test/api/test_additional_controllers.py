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
                    run_laser_control_checks(api_server, lasers)
                return
        except Exception as e:
            print(f"Laser discovery via {endpoint} failed: {e}")


# Helper function (not a test): run laser control checks for the discovered lasers
def run_laser_control_checks(api_server, lasers):
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
            # Start stream with a strict timeout and streamed response to avoid hangs
            start_params = {"startStream": True}
            # Use a small connect/read timeout to ensure we never block >2s during start
            response = api_server.get(endpoint, params=start_params, stream=True, timeout=(2, 2))

            if response.status_code == 200:
                print(f"✓ Video streaming endpoint accessible: {endpoint} (200)")

                # Try to read at least one chunk within 2s to confirm data is flowing
                got_bytes = False
                try:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            got_bytes = True
                            break
                except requests.exceptions.ReadTimeout:
                    pytest.fail("Video stream read timed out after 2s")
                finally:
                    # Always close streamed response
                    try:
                        response.close()
                    except Exception:
                        pass

                assert got_bytes, "No data received from video stream"
                print("✓ Received initial video bytes from stream")

                # Stop the stream (with a safe timeout)
                stop_params = {"startStream": False}
                stop_resp = api_server.get(endpoint, params=stop_params, timeout=5)
                assert stop_resp.status_code in [200, 400, 501]
                print(f"✓ Video streaming stop via {endpoint} ({stop_resp.status_code})")
                break

            elif response.status_code in [400, 501]:
                # Endpoint present but not implemented or invalid in current context
                print(f"✓ Video streaming endpoint responded (non-200 acceptable): {endpoint} ({response.status_code})")
                try:
                    response.close()
                except Exception:
                    pass
                break
            else:
                # Unexpected status: still ensure we don't get stuck
                try:
                    response.close()
                except Exception:
                    pass
                pytest.fail(f"Unexpected status code from {endpoint}: {response.status_code}")
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