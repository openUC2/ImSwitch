"""
API tests for ImSwitch DetectorController endpoints.
Replaces Qt-based liveview tests with FastAPI endpoint testing.
"""
import pytest
import requests
import time
from ..api import api_server, base_url


def test_detector_endpoints_available(api_server):
    """Test that detector API endpoints are accessible."""
    # Test API documentation is available
    response = api_server.get("/docs")
    assert response.status_code == 200
    
    # Test that we can reach the detector endpoints
    response = api_server.get("/DetectorController/getAllDetectorNames")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_detector_names_and_info(api_server):
    """Test getting detector names and information."""
    # Get all detector names
    response = api_server.get("/DetectorController/getAllDetectorNames")
    assert response.status_code == 200
    detector_names = response.json()
    assert len(detector_names) > 0  # Should have at least one detector
    
    # Get info for first detector
    first_detector = detector_names[0]
    response = api_server.get(f"/DetectorController/getDetectorInfo?detectorName={first_detector}")
    assert response.status_code == 200
    info = response.json()
    assert "width" in info
    assert "height" in info


def test_liveview_functionality(api_server):
    """Test liveview start/stop functionality via API (replaces Qt button test)."""
    detector_names = api_server.get("/DetectorController/getAllDetectorNames").json()
    first_detector = detector_names[0]
    
    # Start liveview
    response = api_server.post(f"/DetectorController/startLiveview?detectorName={first_detector}")
    assert response.status_code == 200
    
    # Check if liveview is running
    response = api_server.get(f"/DetectorController/isLiveviewRunning?detectorName={first_detector}")
    assert response.status_code == 200
    is_running = response.json()
    assert is_running is True
    
    # Let liveview run for a moment
    time.sleep(1)
    
    # Stop liveview  
    response = api_server.post(f"/DetectorController/stopLiveview?detectorName={first_detector}")
    assert response.status_code == 200
    
    # Check if liveview stopped
    response = api_server.get(f"/DetectorController/isLiveviewRunning?detectorName={first_detector}")
    assert response.status_code == 200
    is_running = response.json()
    assert is_running is False


def test_detector_parameters(api_server):
    """Test getting and setting detector parameters."""
    detector_names = api_server.get("/DetectorController/getAllDetectorNames").json()
    first_detector = detector_names[0]
    
    # Get current parameters
    response = api_server.get(f"/DetectorController/getDetectorParameters?detectorName={first_detector}")
    assert response.status_code == 200
    params = response.json()
    assert isinstance(params, dict)
    
    # If exposure time parameter exists, try to modify it
    if "exposureTime" in params:
        original_exposure = params["exposureTime"]
        new_exposure = original_exposure * 1.1  # Increase by 10%
        
        response = api_server.put(
            f"/DetectorController/setDetectorParameter",
            json={
                "detectorName": first_detector,
                "parameterName": "exposureTime", 
                "value": new_exposure
            }
        )
        assert response.status_code == 200
        
        # Verify the change
        response = api_server.get(f"/DetectorController/getDetectorParameters?detectorName={first_detector}")
        updated_params = response.json()
        assert abs(updated_params["exposureTime"] - new_exposure) < 0.01


def test_image_capture(api_server):
    """Test image capture functionality."""
    detector_names = api_server.get("/DetectorController/getAllDetectorNames").json()
    first_detector = detector_names[0]
    
    # Capture a single image
    response = api_server.post(f"/DetectorController/captureImage?detectorName={first_detector}")
    assert response.status_code == 200
    
    # The response should contain image data or reference
    result = response.json()
    assert result is not None


@pytest.mark.skip(reason="Requires specific detector setup")
def test_recording_functionality(api_server):
    """Test recording start/stop functionality."""
    detector_names = api_server.get("/DetectorController/getAllDetectorNames").json()
    first_detector = detector_names[0]
    
    # Start recording
    response = api_server.post(f"/DetectorController/startRecording?detectorName={first_detector}")
    assert response.status_code == 200
    
    # Let it record briefly
    time.sleep(2)
    
    # Stop recording
    response = api_server.post(f"/DetectorController/stopRecording?detectorName={first_detector}")
    assert response.status_code == 200


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
