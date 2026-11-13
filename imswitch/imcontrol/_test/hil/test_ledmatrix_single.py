"""One LED of the matrix lights alone, and the cameras see it. SWITCHES LIGHT.

setLEDs lights exactly the LEDs it is given. One LED must change the image by more than the
camera's own noise, and the whole matrix must give a brighter image than that one LED.
The LED is the one closest to the optical axis (from the DPC LED layout), so that it lights the
sample in brightfield; without the DPC controller it is the centre of the setup's Nx x Ny grid.
"""
import base64
import contextlib
import io
import time

import pytest
from PIL import Image, ImageChops, ImageStat

import hil_client as client
from hil_report import attach

SINGLE_INTENSITY = 255         # per colour, of setLEDs (0..255)
MATRIX_INTENSITY = 255         # per colour, of setAllLED, the same as one LED
LIGHT_SETTLE_S = 0.5           # wait after switching a light, so the next frame shows it
AUTO_EXPOSURE_RESET_MS = 1500  # ImSwitch returns to manual exposure after this time
FRAME_RESIZE = 0.25            # the frames are compared at this share of their size
# One LED must change the image by max(MIN_CHANGE, noise * NOISE_FACTOR) grey values.
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


def central_led():
    """Chain index of the LED closest to the optical axis.

    The DPC controller knows every LED's illumination NA; the smallest NA is the most central
    LED. Without it, the middle of the Nx x Ny grid the setup declares (ImSwitch's default
    layout) is taken; its chain order is not known here, so that LED may lie off centre.
    """
    response = client.get("/DPCController/get_led_layout")
    if response.status_code == 200:
        leds = response.json().get("leds") or []
        if leds:
            led = min(leds, key=lambda led: led["na_x"] ** 2 + led["na_y"] ** 2)
            return led["id"], f"smallest illumination NA ({led['na_x']:.3f}, {led['na_y']:.3f})"
    matrix = next(iter(client.setup()["LEDMatrixs"].values()))
    props = matrix.get("managerProperties") or {}
    nx, ny = int(props.get("Nx", 8)), int(props.get("Ny", 8))
    return (ny // 2) * nx + nx // 2, f"middle of the {nx} x {ny} grid (no DPC layout)"


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
                               params={"slot_id": "1", "camera_name": "hil_ledmatrix_single"})
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


def dark_baseline(detector):
    """Measure the camera noise with the scene unchanged.

    Returns (last frame, noise floor, change one LED must exceed).
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

@pytest.fixture(scope="module", autouse=True)
def led_matrix():
    if not client.setup()["LEDMatrixs"]:
        pytest.skip("the setup has no LED matrix")
    if client.get("/LEDMatrixController/setLEDs", params={"ledIds": ""}).status_code == 404:
        pytest.skip("this ImSwitch has no LEDMatrixController/setLEDs")


@pytest.fixture(scope="module", params=["acquisition", "observation"])
def camera(request):
    """A camera that is ready to take frames; with it a test runs once per kind of camera.

    The test sets the exposure itself, with every LED on, so that no frame saturates.
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
            yield detector
    all_lights_off()


@pytest.fixture
def dark():
    """Every light off before and after the test."""
    all_lights_off()
    yield
    all_lights_off()


# --- tests ------------------------------------------------------------------------------------

def test_single_led_is_visible_and_dimmer_than_all(camera, dark):
    led, why = central_led()

    # 1. Every LED on; the exposure is set on this, the brightest scene, so nothing saturates.
    #    The observation camera has no live view for the auto exposure to act on.
    client.get_json("/LEDMatrixController/setAllLED", {
        "intensity_r": MATRIX_INTENSITY, "intensity_g": MATRIX_INTENSITY,
        "intensity_b": MATRIX_INTENSITY})
    time.sleep(LIGHT_SETTLE_S)
    if not is_observation_camera(camera):
        auto_expose(camera)
    every = take_image(camera)

    # 2. Only the one LED: setLEDs switches every other LED off.
    client.get_json("/LEDMatrixController/setLEDs", {
        "ledIds": str(led), "intensity_r": SINGLE_INTENSITY, "intensity_g": SINGLE_INTENSITY,
        "intensity_b": SINGLE_INTENSITY})
    time.sleep(LIGHT_SETTLE_S)
    single = take_image(camera)

    # 3. Dark, at the same exposure: the noise the single LED must exceed.
    all_lights_off()
    unlit, noise_floor, required = dark_baseline(camera)

    attach("dark", unlit)
    attach(f"LED {led} alone", single)
    attach("every LED", every)
    change = image_difference(unlit, single)
    print(f"\nLED {led} ({why}): pixel_change={change:.2f} noise_floor={noise_floor:.2f} "
          f"required={required:.2f}; brightness dark={brightness(unlit):.2f} "
          f"single={brightness(single):.2f} all={brightness(every):.2f}")
    assert change >= required, (
        f"LED {led} alone changed the image only by {change:.2f}; camera noise floor is "
        f"{noise_floor:.2f}, need >= {required:.2f} - setLEDs may not have lit it")
    assert brightness(every) > brightness(single), (
        f"every LED ({brightness(every):.2f}) is not brighter than LED {led} alone "
        f"({brightness(single):.2f}) - setLEDs may have lit more than one LED, or the camera "
        f"saturates")
