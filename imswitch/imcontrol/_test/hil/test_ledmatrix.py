"""The LED matrix can be switched on and off, and every camera sees it go on and off.

SWITCHES LIGHT.

It passes only when every camera of the setup sees both switches.
On: the image changes by more than the camera's own noise.
Off: the image is back to dark, within that noise.
The noise: take a few dark frames and measure how much they differ.
"""
import base64
import contextlib
import io
import time

import pytest
from PIL import Image, ImageChops, ImageStat

import hil_client as client
from hil_report import attach, note

MATRIX_INTENSITY = 500         # per colour, of setAllLED
LIGHT_SETTLE_S = 0.5           # wait after switching a light, so the next frame shows it
AUTO_EXPOSURE_RESET_MS = 1500  # ImSwitch returns to manual exposure after this time
FRAME_RESIZE = 0.25            # the frames are compared at this share of their size
# The matrix must change the image by max(MIN_CHANGE, noise * NOISE_FACTOR) grey values.
# The noise is the largest difference between NOISE_SAMPLES dark frames.
MIN_CHANGE = 1.5
NOISE_FACTOR = 1.5
NOISE_SAMPLES = 6
NOISE_DELAY_S = 0.5


# --- lights -----------------------------------------------------------------------------------

def all_lights_off():
    """Every laser, LED and the LED matrix off, so only the matrix changes the image."""
    for name in client.setup()["lasers"]:
        client.get_json("/LaserController/setLaserValue", {"laserName": name, "value": 0})
        client.get_json("/LaserController/setLaserActive", {"laserName": name, "active": False})
    client.get_json("/LEDMatrixController/setAllLEDOff")
    time.sleep(LIGHT_SETTLE_S)


def set_matrix(on):
    """Every LED of the matrix on at MATRIX_INTENSITY, or off."""
    if on:
        client.get_json("/LEDMatrixController/setAllLED", {
            "intensity_r": MATRIX_INTENSITY, "intensity_g": MATRIX_INTENSITY,
            "intensity_b": MATRIX_INTENSITY})
    else:
        client.get_json("/LEDMatrixController/setAllLEDOff")
    time.sleep(LIGHT_SETTLE_S)


# --- cameras ----------------------------------------------------------------------------------
# An acquisition detector ("forAcquisition": true) gives frames only while its live view runs.
# The observation camera ("forAcquisition": false) gives them at any time.

def is_observation_camera(detector):
    return not client.setup()["detectors"][detector]["forAcquisition"]


def cameras():
    """The cameras that give real frames.
    A camera ImSwitch serves as a mock, or cannot report on, is left out and named in the report.
    """
    found = []
    for detector in client.setup()["detectors"]:
        status = client.get_json("/SettingsController/getCameraStatus", {"detectorName": detector})
        if "error" in status or status.get("status") == "error":
            note(f"{detector} left out: getCameraStatus failed: {status.get('error')}")
        elif status.get("isMock") or str(status.get("model", "")).lower() == "mock":
            note(f"{detector} left out: ImSwitch served a mock camera")
        else:
            found.append(detector)
    return found


def take_image(detector):
    """One small greyscale frame (PIL) of either kind of camera."""
    if is_observation_camera(detector):
        # camera_name only names the folder ImSwitch saves the frame to.
        response = client.post("/ExperimentController/snapOverviewImage", timeout=60,
                               params={"slot_id": "1", "camera_name": "hil_ledmatrix"})
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


def brightness(image):
    """Mean grey value of a frame."""
    return ImageStat.Stat(image).mean[0]


def exposure_and_gain(detector):
    """ "exposure 20, gain 0" as the camera reports them; None for a value it does not report."""
    values = [client.get_json("/SettingsController/getDetectorParameter",
                              {"detectorName": detector, "parameterName": name})
              for name in ("exposure", "gain")]
    return f"exposure {values[0]}, gain {values[1]}"


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


@contextlib.contextmanager
def ready(detector):
    """The camera ready to take frames inside a ``with`` block.

    An acquisition detector streams, and its exposure is set once, in the dark,
    so that the dark, the lit and the switched-off frames share it.
    The observation camera is left alone: it has no live view for the auto exposure to act on.
    """
    if is_observation_camera(detector):
        yield
        return
    with live_view(detector):
        auto_expose(detector)
        yield


def sees_switching(detector, switch):
    """Switch the light on and off in front of one camera; ``switch(on)`` switches it.

    Returns what the camera did not see, as lines; empty when it saw both.
    """
    with ready(detector):
        dark, noise_floor, required = dark_baseline(detector)
        switch(True)
        lit = take_image(detector)
        switch(False)
        off = take_image(detector)
        settings = exposure_and_gain(detector)
    attach(f"{detector}: dark", dark)
    attach(f"{detector}: on", lit)
    attach(f"{detector}: off again", off)
    on_change, off_change = image_difference(dark, lit), image_difference(dark, off)
    print(f"\n{detector}: on changed {on_change:.2f}, off is {off_change:.2f} from dark, "
          f"noise floor {noise_floor:.2f}, required {required:.2f}; brightness dark "
          f"{brightness(dark):.2f}, on {brightness(lit):.2f}; {settings}")
    missed = []
    if on_change < required:
        missed.append(f"{detector}: on changed the image only by {on_change:.2f}, "
                      f"need >= {required:.2f} (noise floor {noise_floor:.2f}; {settings})")
    if off_change >= required:
        missed.append(f"{detector}: off is still {off_change:.2f} from dark, "
                      f"need < {required:.2f}")
    return missed


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def led_matrix():
    if not client.setup()["LEDMatrixs"]:
        pytest.skip("the setup has no LED matrix")


@pytest.fixture
def dark():
    """Every light off before and after the test."""
    all_lights_off()
    yield
    all_lights_off()


# --- tests ------------------------------------------------------------------------------------

def test_all_leds_off_is_accepted():
    client.get_json("/LEDMatrixController/setAllLEDOff", timeout=5)


def test_every_camera_sees_the_matrix_switch_on_and_off(dark):
    found = cameras()
    if not found:
        pytest.skip("the setup has no camera that gives real frames")
    missed = []
    # Every camera is measured before the verdict, so the report has the photos of all of them.
    for detector in found:
        missed += sees_switching(detector, set_matrix)
    assert not missed, f"LED matrix @ {MATRIX_INTENSITY}:\n" + "\n".join(missed)
