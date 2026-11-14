"""Every configured camera returns a frame, at the resolution it reports. Only reads.

ImSwitch reads the two kinds of camera in different ways:
- an acquisition detector ("forAcquisition": true in the setup) with snapNumpyToFastAPI, which
  only works while its live view runs; the runner (__main__.py) has started it;
- the observation camera ("forAcquisition": false) with snapOverviewImage, at any time.
"""
import base64
import io

import pytest
from PIL import Image

import hil_client as client
from hil_report import attach

SNAP_RESIZE = 0.1  # resizeFactor of the acquisition frame, to keep the transfer small


def detectors():
    return list(client.setup()["detectors"])


def is_observation_camera(detector):
    return not client.setup()["detectors"][detector]["forAcquisition"]


def camera_status(detector):
    """getCameraStatus; skips the test if the camera has an error or is ImSwitch's mock camera."""
    status = client.get_json("/SettingsController/getCameraStatus", {"detectorName": detector})
    if "error" in status or status.get("status") == "error":
        pytest.skip(f"{detector}: getCameraStatus failed: {status.get('error')}")
    if status.get("isMock") or str(status.get("model", "")).lower() == "mock":
        pytest.skip(f"{detector}: ImSwitch served a mock camera (model={status.get('model')!r})")
    return status


def snap(detector):
    """One frame as (PIL image, the resize factor ImSwitch applied to it)."""
    if is_observation_camera(detector):
        # camera_name only names the folder ImSwitch saves the frame to.
        body = client.post_json("/ExperimentController/snapOverviewImage", timeout=10,
                                params={"slot_id": "1", "camera_name": "hil_camera_capture"})
        frame = Image.open(io.BytesIO(base64.b64decode(body["imageBase64"])))
        assert frame.format == "JPEG", f"snapOverviewImage returned {frame.format}"
        return frame, 1.0
    response = client.get("/RecordingController/snapNumpyToFastAPI", timeout=10, params={
        "detectorName": detector, "resizeFactor": SNAP_RESIZE})
    assert response.status_code == 200, response.text
    frame = Image.open(io.BytesIO(response.content))
    assert frame.format == "PNG", f"snapNumpyToFastAPI returned {frame.format}"
    return frame, SNAP_RESIZE


every_detector = client.parametrize_from_rig("detector", detectors, "the setup has no detectors")


@every_detector
def test_camera_returns_image(detector):
    camera_status(detector)
    frame, _ = snap(detector)
    attach(detector, frame)
    assert frame.width > 0 and frame.height > 0


@every_detector
def test_camera_matches_sensor_resolution(detector):
    status = camera_status(detector)
    width = status.get("currentWidth") or status.get("sensorWidth")
    height = status.get("currentHeight") or status.get("sensorHeight")
    if not width or not height:
        pytest.skip(f"{detector}: getCameraStatus reports no frame size ({width}x{height})")
    frame, resize = snap(detector)
    expected = (int(width * resize), int(height * resize))
    assert frame.size == expected, (
        f"{detector}: got {frame.width}x{frame.height}, expected {expected[0]}x{expected[1]} "
        f"({width}x{height} at resizeFactor {resize})")
