"""
API tests for ImSwitch PositionerController endpoints.
Tests motor/stage movement functionality via REST API.
"""
import pytest
import requests
import time
from ..api import api_server, base_url


def test_positioner_endpoints_available(api_server):
    """Test that positioner API endpoints are accessible."""
    response = api_server.get("/PositionerController/getPositioners")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_positioner_info(api_server):
    """Test getting positioner information."""
    # Get all positioners
    response = api_server.get("/PositionerController/getPositioners")
    assert response.status_code == 200
    positioners = response.json()
    assert len(positioners) > 0
    
    # Get info for each positioner
    for positioner_name in positioners:
        response = api_server.get(f"/PositionerController/getPosition?positionerName={positioner_name}")
        assert response.status_code == 200
        position = response.json()
        assert isinstance(position, dict)


def test_position_movement(api_server):
    """Test moving positioners to specific positions."""
    # Get available positioners
    response = api_server.get("/PositionerController/getPositioners")
    positioners = response.json()
    
    if not positioners:
        pytest.skip("No positioners available for testing")
    
    first_positioner = list(positioners.keys())[0]
    
    # Get current position
    response = api_server.get(f"/PositionerController/getPosition?positionerName={first_positioner}")
    assert response.status_code == 200
    original_position = response.json()
    
    # Move to a new position (small relative move)
    new_position = {}
    for axis, current_pos in original_position.items():
        # Make a small move (1 unit) to avoid hitting limits
        new_position[axis] = current_pos + 1
    
    response = api_server.post(
        f"/PositionerController/setPosition",
        json={
            "positionerName": first_positioner,
            "position": new_position
        }
    )
    assert response.status_code == 200
    
    # Give time for movement
    time.sleep(0.5)
    
    # Check new position
    response = api_server.get(f"/PositionerController/getPosition?positionerName={first_positioner}")
    assert response.status_code == 200
    updated_position = response.json()
    
    # Verify movement occurred (for virtual stage, should be exact)
    for axis in new_position:
        assert abs(updated_position[axis] - new_position[axis]) < 0.1


def test_relative_movement(api_server):
    """Test relative position movements."""
    response = api_server.get("/PositionerController/getPositioners")
    positioners = response.json()
    
    if not positioners:
        pytest.skip("No positioners available for testing")
    
    first_positioner = list(positioners.keys())[0]
    
    # Get current position
    response = api_server.get(f"/PositionerController/getPosition?positionerName={first_positioner}")
    original_position = response.json()
    
    # Make relative move
    relative_move = {"X": 2, "Y": -1}  # Move 2 units in X, -1 in Y
    
    response = api_server.post(
        f"/PositionerController/moveRelative",
        json={
            "positionerName": first_positioner,
            "relativeMove": relative_move
        }
    )
    assert response.status_code == 200
    
    time.sleep(0.5)
    
    # Check final position
    response = api_server.get(f"/PositionerController/getPosition?positionerName={first_positioner}")
    final_position = response.json()
    
    # Verify relative movement
    for axis, relative_amount in relative_move.items():
        if axis in original_position and axis in final_position:
            expected = original_position[axis] + relative_amount
            assert abs(final_position[axis] - expected) < 0.1


def test_positioner_speed_control(api_server):
    """Test setting positioner movement speed."""
    response = api_server.get("/PositionerController/getPositioners")
    positioners = response.json()
    
    if not positioners:
        pytest.skip("No positioners available for testing")
    
    first_positioner = list(positioners.keys())[0]
    
    # Try to set speed (may not be supported by all positioners)
    response = api_server.post(
        f"/PositionerController/setSpeed",
        json={
            "positionerName": first_positioner,
            "speed": {"X": 5000, "Y": 5000}
        }
    )
    # Don't assert success as not all positioners support speed control
    # Just check it doesn't crash
    assert response.status_code in [200, 400, 404]


@pytest.mark.skip(reason="Requires homing capability")
def test_homing_functionality(api_server):
    """Test homing functionality if available."""
    response = api_server.get("/PositionerController/getPositioners")
    positioners = response.json()
    
    if not positioners:
        pytest.skip("No positioners available for testing")
    
    first_positioner = list(positioners.keys())[0]
    
    # Attempt to home (may not be supported)
    response = api_server.post(f"/PositionerController/home?positionerName={first_positioner}")
    # Accept various status codes as homing may not be implemented
    assert response.status_code in [200, 400, 404, 501]


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
