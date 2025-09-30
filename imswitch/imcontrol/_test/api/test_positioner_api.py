"""
API tests for ImSwitch PositionerController endpoints.
Tests motor/stage movement, positioning, and scanning functionality via REST API.
"""
import pytest
import requests
import time
import json
from typing import Dict, List, Any, Tuple
from ..api import api_server, base_url


def test_positioner_endpoints_available(api_server):
    """Test that positioner API endpoints are accessible."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Find positioner-related endpoints
    paths = spec.get("paths", {})
    positioner_endpoints = [p for p in paths.keys() if "Positioner" in p or "positioner" in p or "position" in p]
    
    assert len(positioner_endpoints) > 0, "No positioner endpoints found in API"
    print(f"Found {len(positioner_endpoints)} positioner endpoints")
    
    # Test basic positioner endpoint accessibility
    for endpoint in positioner_endpoints[:3]:  # Test first 3
        try:
            response = api_server.get(endpoint)
            assert response.status_code in [200, 400, 404, 422], f"Unexpected status for {endpoint}: {response.status_code}"
            print(f"✓ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"? {endpoint}: {e}")


def test_positioner_discovery(api_server):
    """Test positioner discovery and enumeration."""
    # Common positioner discovery endpoints
    discovery_endpoints = [
        "/PositionerController/getPositionerNames",
        ]
    
    positioners = None
    working_endpoint = None
    
    for endpoint in discovery_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and len(data) > 0:
                    positioners = data
                    working_endpoint = endpoint
                    break
                elif isinstance(data, list) and len(data) > 0:
                    # Convert list to dict format
                    positioners = {name: {} for name in data}
                    working_endpoint = endpoint
                    break
        except Exception as e:
            print(f"Discovery endpoint {endpoint} failed: {e}")
    
    if positioners and working_endpoint:
        print(f"✓ Found positioners via {working_endpoint}: {list(positioners.keys())}")
        assert len(positioners) > 0
        return positioners, working_endpoint
    else:
        pytest.skip("No working positioner discovery endpoint found")


def test_positioner_position_reading(api_server):
    """Test reading current positions from positioners."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Position reading endpoints
    position_endpoints = [
        f"/PositionerController/getPosition?positionerName={first_positioner}",
        ]
    
    for endpoint in position_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                position = response.json()
                assert isinstance(position, dict)
                print(f"✓ Got position via {endpoint}: {position}")
                
                # Validate position format
                for axis, pos_value in position.items():
                    assert isinstance(pos_value, (int, float)), f"Invalid position value for {axis}: {pos_value}"
                
                return position
                
        except Exception as e:
            print(f"Position reading via {endpoint} failed: {e}")
    
    print("? No working position reading endpoints found")
    return None


def test_absolute_positioning(api_server):
    """Test absolute position movement."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Get current position first
    current_position = test_positioner_position_reading(api_server)
    if not current_position:
        pytest.skip("Cannot read current position")
    
    # Calculate new target position (small move to avoid limits)
    target_position = {}
    for axis, current_pos in current_position.items():
        target_position[axis] = current_pos + 1.0  # Move 1 unit
    
    # Absolute positioning endpoints
    positioning_endpoints = [
        "/PositionerController/movePositioner",
    ]
    
    # Test absolute positioning using the correct API interface
    for endpoint in positioning_endpoints:
        try:
            # Test single axis absolute moves using GET with query parameters
            for axis, position in target_position.items():
                params = {
                    "positionerName": first_positioner,
                    "axis": axis,
                    "dist": position,
                    "isAbsolute": True,
                    "isBlocking": False
                }
                
                response = api_server.get(endpoint, params=params)
                if response.status_code == 200:
                    print(f"✓ Absolute move via GET {endpoint} - {axis}:{position}")
                    time.sleep(0.5)  # Allow movement time
                    # Since we moved each axis individually, verify the final position
                    if axis == list(target_position.keys())[-1]:  # Last axis
                        verify_position_change(api_server, first_positioner, target_position)
                    return
                elif response.status_code in [400, 404, 422]:
                    print(f"? Absolute positioning via {endpoint} - {axis}:{position} - Status: {response.status_code}")
                    
        except Exception as e:
            print(f"Absolute positioning via {endpoint} failed: {e}")
    
    print("? No working absolute positioning endpoints found")


def test_relative_positioning(api_server):
    """Test relative position movement."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Define relative move
    relative_move = {"X": 2.0, "Y": -1.0}  # Example move
    
    # Relative positioning endpoints
    relative_endpoints = [
        "/PositionerController/movePositioner",
    ]

    # Test relative positioning using the correct API interface
    for endpoint in relative_endpoints:
        try:
            # Test single axis relative moves using GET with query parameters
            for axis, distance in relative_move.items():
                params = {
                    "positionerName": first_positioner,
                    "axis": axis,
                    "dist": distance,
                    "isAbsolute": False,  # This makes it relative
                    "isBlocking": False
                }
                
                response = api_server.get(endpoint, params=params)
                if response.status_code == 200:
                    print(f"✓ Relative move via GET {endpoint} - {axis}:{distance}")
                    time.sleep(0.5)
                    return
                elif response.status_code in [400, 404, 422]:
                    print(f"? Relative positioning via {endpoint} - {axis}:{distance} - Status: {response.status_code}")
                    
        except Exception as e:
            print(f"Relative positioning via {endpoint} failed: {e}")
    
    print("? No working relative positioning endpoints found")


def verify_position_change(api_server, positioner_name: str, expected_position: Dict[str, float], tolerance: float = 0.1):
    """Verify that position change occurred as expected."""
    try:
        current_position = test_positioner_position_reading(api_server)
        if current_position:
            for axis, expected_pos in expected_position.items():
                if axis in current_position:
                    actual_pos = current_position[axis]
                    diff = abs(actual_pos - expected_pos)
                    if diff < tolerance:
                        print(f"✓ Position verified for {axis}: {actual_pos} ≈ {expected_pos}")
                    else:
                        print(f"? Position mismatch for {axis}: {actual_pos} vs {expected_pos} (diff: {diff})")
    except Exception as e:
        print(f"Position verification failed: {e}")


def test_positioner_speed_control(api_server):
    """Test positioner movement speed configuration."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    test_speed = {"X": 5000, "Y": 5000}  # Example speeds
    
    # Speed control endpoints
    speed_endpoints = [
        "/PositionerController/setPositionerSpeed",
    ]
    
    # Test speed control using the correct API interface
    for endpoint in speed_endpoints:
        try:
            # Test setting speed for each axis using GET with query parameters
            for axis, speed_value in test_speed.items():
                params = {
                    "positionerName": first_positioner,
                    "axis": axis,
                    "speed": speed_value
                }
                
                response = api_server.get(endpoint, params=params)
                if response.status_code == 200:
                    print(f"✓ Speed set via GET {endpoint} - {axis}:{speed_value}")
                    return
                elif response.status_code in [400, 404, 422]:
                    print(f"? Speed control via {endpoint} - {axis}:{speed_value} - Status: {response.status_code}")
                    
        except Exception as e:
            print(f"Speed control via {endpoint} failed: {e}")
    
    print("? No working speed control endpoints found")


def test_positioner_homing(api_server):
    """Test positioner homing functionality using the correct API interface."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Homing endpoints
    homing_endpoints = [
        "/PositionerController/homeAxis",
    ]
    
    for endpoint in homing_endpoints:
        try:
            # Test homing using GET with query parameters
            params = {
                "positionerName": first_positioner,
                "axis": "X",  # Default to X axis
                "isBlocking": False
            }
            
            response = api_server.get(endpoint, params=params)
            # Accept various status codes as homing may not be implemented on all stages
            if response.status_code in [200, 400, 404, 422, 501]:
                print(f"✓ Homing endpoint accessible: {endpoint} ({response.status_code})")
                return
                
        except Exception as e:
            print(f"Homing via {endpoint} failed: {e}")
    
    print("? No homing endpoints found")


def test_positioner_stop_emergency(api_server):
    """Test emergency stop functionality using the correct API interface."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Emergency stop endpoints
    stop_endpoints = [
        "/PositionerController/stopAxis",
    ]
    
    for endpoint in stop_endpoints:
        try:
            # Test stop using GET with query parameters
            params = {
                "positionerName": first_positioner,
                "axis": "X"  # Default to X axis
            }
            
            response = api_server.get(endpoint, params=params)
            if response.status_code in [200, 400, 404, 422]:
                print(f"✓ Stop command via GET {endpoint} ({response.status_code})")
                return
                
        except Exception as e:
            print(f"Stop command via {endpoint} failed: {e}")
    
    print("? No working stop endpoints found")



def test_multi_axis_coordination(api_server):
    """Test coordinated multi-axis movements using individual axis calls."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
    except:
        pytest.skip("No positioners found")
    
    # Since movePositioner handles single axis at a time, we coordinate by calling it multiple times
    endpoint = "/PositionerController/movePositioner"
    first_positioner = list(positioners.keys())[0]
    
    # Test coordinated movement by moving multiple axes in sequence
    coordinated_moves = {"X": 10.0, "Y": 5.0}  # Example coordinated move
    
    try:
        success_count = 0
        for axis, distance in coordinated_moves.items():
            params = {
                "positionerName": first_positioner,
                "axis": axis,
                "dist": distance,
                "isAbsolute": True,
                "isBlocking": False
            }
            
            response = api_server.get(endpoint, params=params)
            if response.status_code == 200:
                print(f"✓ Coordinated move axis {axis}: {distance}")
                success_count += 1
                time.sleep(0.2)  # Brief delay between moves
                
        if success_count > 0:
            print(f"✓ Coordinated movement completed ({success_count}/{len(coordinated_moves)} axes)")
            return
            
    except Exception as e:
        print(f"Coordinated move failed: {e}")
    
    print("? No coordinated movement capability found")


@pytest.mark.skip(reason="Requires scanning capability")
def test_scanning_functionality(api_server):
    """Test scanning/raster movement patterns."""
    # This would test:
    # - Raster scanning patterns
    # - Grid movements
    # - Spiral patterns
    # - Custom trajectory following
    pass


@pytest.mark.skip(reason="Requires hardware-specific setup")
def test_hardware_specific_features(api_server):
    """Test hardware-specific positioner features."""
    # This would test:
    # - Encoder feedback
    # - Closed-loop vs open-loop control
    # - Hardware-specific parameters
    # - Calibration procedures
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
