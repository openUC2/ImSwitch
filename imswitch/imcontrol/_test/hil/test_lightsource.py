"""Every laser/LED switches through the API, and every LED reaches the cameras. SWITCHES LIGHT.

A light counts as visible when it changes the image by more than the camera's own noise:
take a few dark frames, measure how much they differ, then compare a dark and a lit frame.
"""
import base64
import contextlib
import io
import time

import pytest
from PIL import Image, ImageChops, ImageStat

import hil_client as client
from hil_report import attach

LIGHT_SETTLE_S = 0.5           # wait after switching a light, so the next frame shows it
AUTO_EXPOSURE_RESET_MS = 1500  # ImSwitch returns to manual exposure after this time
FRAME_RESIZE = 0.25            # the frames are compared at this share of their size
# A light must change the image by max(MIN_CHANGE, noise * NOISE_FACTOR) grey values.
# The noise is the largest difference between NOISE_SAMPLES dark frames.
MIN_CHANGE = 1.5
NOISE_FACTOR = 1.5
NOISE_SAMPLES = 6
NOISE_DELAY_S = 0.5


# --- lights -----------------------------------------------------------------------------------

def light_sources():
    """Names of every laser and LED of the setup."""
    return list(client.setup()["lasers"])


def leds():
    """Names of the LEDs. The lasers are left out: on these rigs their beam misses the sensor.

    The setup lists lasers and LEDs alike under "lasers", so only the name tells them apart.
    """
    return [name for name in light_sources()
            if "led" in name.lower() and "laser" not in name.lower()]


def set_light(name, on):
    """Switch one laser/LED on (at the highest value its setup allows) or off."""
    value = client.setup()["lasers"][name]["valueRangeMax"] if on else 0
    client.get_json("/LaserController/setLaserValue", {"laserName": name, "value": value})
    client.get_json("/LaserController/setLaserActive", {"laserName": name, "active": on})
    time.sleep(LIGHT_SETTLE_S)


def all_lights_off():
    """Every laser, LED and the LED matrix off, so only the light under test changes the image."""
    for name in light_sources():
        client.get_json("/LaserController/setLaserValue", {"laserName": name, "value": 0})
        client.get_json("/LaserController/setLaserActive", {"laserName": name, "active": False})
    if client.setup()["LEDMatrixs"]:
        client.get_json("/LEDMatrixController/setAllLEDOff")
    time.sleep(LIGHT_SETTLE_S)


# --- cameras ----------------------------------------------------------------------------------
# An acquisition detector ("forAcquisition": true in the setup) gives frames only while its
# live view runs. The observation camera ("forAcquisition": false) gives them at any time.

def is_observation_camera(detector):
    return not client.setup()["detectors"][detector]["forAcquisition"]


def take_image(detector):
    """One small greyscale frame (PIL) of either kind of camera."""
    if is_observation_camera(detector):
        # camera_name only names the folder ImSwitch saves the frame to.
        response = client.post("/ExperimentController/snapOverviewImage", timeout=60,
                               params={"slot_id": "1", "camera_name": "hil_lightsource"})
        if response.status_code == 400:
            pytest.skip(f"observation camera not connected: {response.text}")
        raw = base64.b64decode(client.json_of(response)["imageBase64"])
        frame = Image.open(io.BytesIO(raw)).convert("L")
        return frame.resize((max(1, int(frame.width * FRAME_RESIZE)),
                             max(1, int(frame.height * FRAME_RESIZE))))
    response = client.get("/RecordingController/snapNumpyToFastAPI", timeout=60,
                          params={"detectorName": detector, "resizeFactor": FRAME_RESIZE})
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content)).convert("L")


def image_difference(first, second):
    """Mean absolute grey-value difference between two frames."""
    return ImageStat.Stat(ImageChops.difference(first, second)).mean[0]


def dark_baseline(detector):
    """Measure the camera noise with the scene unchanged.

    Returns (last frame, noise floor, change a light must exceed).
    """
    previous = take_image(detector)
    drifts = []
    for _ in range(NOISE_SAMPLES - 1):
        time.sleep(NOISE_DELAY_S)
        current = take_image(detector)
        drifts.append(image_difference(previous, current))
        previous = current
    noise_floor = max(drifts)
    required = max(MIN_CHANGE, noise_floor * NOISE_FACTOR)
    print(f"\n{detector} dark noise: drifts={[round(d, 2) for d in drifts]}, "
          f"noise_floor={noise_floor:.2f}, required_change={required:.2f}")
    return previous, noise_floor, required


@contextlib.contextmanager
def live_view(detector):
    """Keep an acquisition detector streaming inside a ``with`` block.

    Only a live view that was started here is stopped again.
    """
    # The HTTP status is 200 even when ImSwitch refuses, so only the "status" field tells.
    status = client.post_json("/LiveViewController/startLiveView", {"detectorName": detector},
                              timeout=60).get("status")
    if status == "long_exposure":
        pytest.skip(f"{detector}: exposure too long for live view")
    assert status in ("success", "already_running"), (
        f"{detector}: startLiveView answered {status!r}")
    time.sleep(0.4)
    try:
        yield
    finally:
        if status == "success":
            client.get_json("/LiveViewController/stopLiveView", {"detectorName": detector})


def auto_expose(detector):
    """One auto-exposure pass on the scene as it is now; needs a running live view."""
    client.get_json("/SettingsController/setDetectorExposureOnce",
                    {"detectorName": detector, "resetDelayMs": AUTO_EXPOSURE_RESET_MS})
    time.sleep(AUTO_EXPOSURE_RESET_MS / 1000 + 0.1)


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["acquisition", "observation"])
def camera(request):
    """A camera that is ready to take frames; with it a test runs once per kind of camera.

    The exposure is set once, in the dark, so that the dark and the lit frames share it. The
    observation camera is left alone: it has no live view for the auto exposure to act on.
    """
    wanted = request.param == "observation"
    names = [name for name in client.setup()["detectors"] if is_observation_camera(name) == wanted]
    if not names:
        pytest.skip(f"the setup has no {request.param} camera")
    detector = names[0]
    status = client.get_json("/SettingsController/getCameraStatus", {"detectorName": detector})
    if "error" in status or status.get("status") == "error":
        pytest.skip(f"{detector}: getCameraStatus failed: {status.get('error')}")
    if status.get("isMock") or str(status.get("model", "")).lower() == "mock":
        pytest.skip(f"{detector}: ImSwitch served a mock camera")

    all_lights_off()
    if wanted:
        yield detector
    else:
        with live_view(detector):
            auto_expose(detector)
            yield detector
    all_lights_off()


@pytest.fixture
def dark():
    """Every light off before and after the test."""
    all_lights_off()
    yield
    all_lights_off()


# --- tests ------------------------------------------------------------------------------------

@client.parametrize_from_rig("name", light_sources, "the setup has no lasers/LEDs")
def test_lightsource_reports_active(name):
    """ImSwitch's own state follows the commands (at value 1); this does not prove light."""
    try:
        client.get_json("/LaserController/setLaserValue", {"laserName": name, "value": 1})
        client.get_json("/LaserController/setLaserActive", {"laserName": name, "active": True})
        active = client.get_json("/LaserController/getLaserActive", {"laserName": name})
        value = client.get_json("/LaserController/getLaserValue", {"laserName": name})
        assert active is True, f"{name}: did not become active"
        assert (value or 0) > 0, f"{name}: value is not positive"

        client.get_json("/LaserController/setLaserActive", {"laserName": name, "active": False})
        active = client.get_json("/LaserController/getLaserActive", {"laserName": name})
        assert active is False, f"{name}: did not become inactive"
    finally:
        set_light(name, False)


@client.parametrize_from_rig("name", leds, "no LED among the light sources of the setup")
def test_light_source_is_visible_to_camera(name, camera, dark):
    unlit, noise_floor, required = dark_baseline(camera)

    set_light(name, True)
    lit = take_image(camera)

    attach("dark", unlit)
    attach("bright", lit)
    change = image_difference(unlit, lit)
    print(f"\n{name}: pixel_change={change:.2f} noise_floor={noise_floor:.2f} "
          f"required={required:.2f}")
    assert lit.getextrema()[1] > 0, f"{name} is on but every pixel is still 0"
    assert change >= required, (f"{name}: image changed only by {change:.2f}; camera noise floor "
                                f"is {noise_floor:.2f}, need >= {required:.2f}")
