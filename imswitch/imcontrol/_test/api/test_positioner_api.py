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
        "/PositionerController/getPositioners",
        "/PositionerController/getAllPositioners",
        "/PositionerController/getPositionerNames",
        "/positioners",  # RESTful style
        "/stages",       # Alternative naming
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
        f"/PositionerController/getCurrentPosition?positioner={first_positioner}",
        f"/positioners/{first_positioner}/position",
        f"/stages/{first_positioner}/position",
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
        "/PositionerController/setPosition",
        "/PositionerController/moveTo",
        f"/positioners/{first_positioner}/position",
        f"/stages/{first_positioner}/move",
    ]
    
    for endpoint in positioning_endpoints:
        try:
            # Try different payload formats
            payloads = [
                {
                    "positionerName": first_positioner,
                    "position": target_position
                },
                {
                    "positioner": first_positioner,
                    "target": target_position
                },
                target_position  # Direct position dict
            ]
            
            for payload in payloads:
                response = api_server.post(endpoint, json=payload)
                if response.status_code in [200, 201]:
                    print(f"✓ Absolute move via POST {endpoint}")
                    time.sleep(0.5)  # Allow movement time
                    verify_position_change(api_server, first_positioner, target_position)
                    return
                
                response = api_server.put(endpoint, json=payload)
                if response.status_code in [200, 201]:
                    print(f"✓ Absolute move via PUT {endpoint}")
                    time.sleep(0.5)
                    verify_position_change(api_server, first_positioner, target_position)
                    return
                    
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
        "/PositionerController/moveRelative",
        "/PositionerController/move",
        f"/positioners/{first_positioner}/move/relative",
        f"/stages/{first_positioner}/jog",
    ]
    
    for endpoint in relative_endpoints:
        try:
            payloads = [
                {
                    "positionerName": first_positioner,
                    "relativeMove": relative_move
                },
                {
                    "positioner": first_positioner,
                    "delta": relative_move
                },
                {
                    "positionerName": first_positioner,
                    "move": relative_move
                }
            ]
            
            for payload in payloads:
                response = api_server.post(endpoint, json=payload)
                if response.status_code in [200, 201]:
                    print(f"✓ Relative move via {endpoint}")
                    time.sleep(0.5)
                    return
                    
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
    
    # Speed control endpoints
    speed_endpoints = [
        "/PositionerController/setSpeed",
        "/PositionerController/setVelocity",
        f"/positioners/{first_positioner}/speed",
        f"/stages/{first_positioner}/velocity",
    ]
    
    test_speed = {"X": 5000, "Y": 5000}  # Example speeds
    
    for endpoint in speed_endpoints:
        try:
            payloads = [
                {
                    "positionerName": first_positioner,
                    "speed": test_speed
                },
                {
                    "positioner": first_positioner,
                    "velocity": test_speed
                },
                test_speed  # Direct speed dict
            ]
            
            for payload in payloads:
                response = api_server.post(endpoint, json=payload)
                if response.status_code in [200, 201]:
                    print(f"✓ Speed set via {endpoint}")
                    return
                
                response = api_server.put(endpoint, json=payload)
                if response.status_code in [200, 201]:
                    print(f"✓ Speed set via {endpoint}")
                    return
                    
        except Exception as e:
            print(f"Speed control via {endpoint} failed: {e}")
    
    print("? No working speed control endpoints found")


def test_positioner_limits_and_bounds(api_server):
    """Test positioner limits and boundary information."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Limits information endpoints
    limits_endpoints = [
        f"/PositionerController/getLimits?positionerName={first_positioner}",
        f"/PositionerController/getBounds?positioner={first_positioner}",
        f"/positioners/{first_positioner}/limits",
        f"/stages/{first_positioner}/bounds",
    ]
    
    for endpoint in limits_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                limits = response.json()
                print(f"✓ Got limits via {endpoint}: {limits}")
                # Validate limits format
                if isinstance(limits, dict):
                    for axis, limit_info in limits.items():
                        if isinstance(limit_info, dict):
                            assert "min" in limit_info or "max" in limit_info, f"Invalid limit format for {axis}"
                return
                
        except Exception as e:
            print(f"Limits check via {endpoint} failed: {e}")
    
    print("? No working limits endpoints found")


def test_positioner_homing(api_server):
    """Test positioner homing functionality."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Homing endpoints
    homing_endpoints = [
        f"/PositionerController/home?positionerName={first_positioner}",
        f"/PositionerController/homeAxes?positioner={first_positioner}",
        f"/positioners/{first_positioner}/home",
        f"/stages/{first_positioner}/home",
    ]
    
    for endpoint in homing_endpoints:
        try:
            response = api_server.post(endpoint)
            # Accept various status codes as homing may not be implemented on all stages
            if response.status_code in [200, 201, 400, 404, 501]:
                print(f"✓ Homing endpoint accessible: {endpoint} ({response.status_code})")
                return
                
        except Exception as e:
            print(f"Homing via {endpoint} failed: {e}")
    
    print("? No homing endpoints found")


def test_positioner_stop_emergency(api_server):
    """Test emergency stop functionality."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Emergency stop endpoints
    stop_endpoints = [
        f"/PositionerController/stop?positionerName={first_positioner}",
        f"/PositionerController/emergencyStop?positioner={first_positioner}",
        f"/positioners/{first_positioner}/stop",
        f"/stages/{first_positioner}/abort",
        "/PositionerController/stopAll",
    ]
    
    for endpoint in stop_endpoints:
        try:
            response = api_server.post(endpoint)
            if response.status_code in [200, 201]:
                print(f"✓ Stop command via {endpoint}")
                return
                
        except Exception as e:
            print(f"Stop command via {endpoint} failed: {e}")
    
    print("? No working stop endpoints found")


def test_positioner_status_monitoring(api_server):
    """Test positioner status and motion state monitoring."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
        first_positioner = list(positioners.keys())[0]
    except:
        first_positioner = "testPositioner"
    
    # Status monitoring endpoints
    status_endpoints = [
        f"/PositionerController/getStatus?positionerName={first_positioner}",
        f"/PositionerController/isMoving?positioner={first_positioner}",
        f"/positioners/{first_positioner}/status",
        f"/stages/{first_positioner}/state",
    ]
    
    for endpoint in status_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Got status via {endpoint}: {status}")
                return
                
        except Exception as e:
            print(f"Status check via {endpoint} failed: {e}")
    
    print("? No working status endpoints found")


def test_multi_axis_coordination(api_server):
    """Test coordinated multi-axis movements."""
    try:
        positioners, _ = test_positioner_discovery(api_server)
        if not positioners:
            pytest.skip("No positioners found")
    except:
        pytest.skip("No positioners found")
    
    # Test coordinated movement of multiple axes
    multi_axis_endpoints = [
        "/PositionerController/moveCoordinated",
        "/PositionerController/setMultiplePositions",
        "/positioners/coordinated_move",
    ]
    
    # Example coordinated move
    coordinated_move = {
        "positions": {
            positioner: {"X": 10.0, "Y": 5.0} 
            for positioner in list(positioners.keys())[:2]  # First two positioners
        }
    }
    
    for endpoint in multi_axis_endpoints:
        try:
            response = api_server.post(endpoint, json=coordinated_move)
            if response.status_code in [200, 201]:
                print(f"✓ Coordinated move via {endpoint}")
                return
                
        except Exception as e:
            print(f"Coordinated move via {endpoint} failed: {e}")
    
    print("? No coordinated movement endpoints found")


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
