"""One LED of the matrix can be lit on its own, and every camera sees it go on and off.
It is dimmer than the whole matrix.

SWITCHES LIGHT.

It passes only when every camera of the setup sees both switches.
setLEDs lights exactly the LEDs it is given.
On: one LED changes the image by more than the camera's own noise.
Off: the image is closer to dark than to the lit image.
Only differences between frames are judged, never a brightness by itself.
The whole matrix must change the image more than that one LED.
The LED is the one closest to the optical axis (from the DPC LED layout),
so that it lights the sample in brightfield;
without the DPC controller it is the centre of the setup's Nx x Ny grid.
"""
import base64
import contextlib
import io
import time

import pytest
from PIL import Image, ImageChops, ImageStat

import hil_client as client
from hil_report import attach, note

SINGLE_INTENSITY = 255         # per colour, of setLEDs (0..255)
MATRIX_INTENSITY = 255         # per colour, of setAllLED (0..255), the same as one LED
# Wait after switching a light, before the next frame.
# The observation camera sets its exposure by itself and needs more than a second for it.
LIGHT_SETTLE_S = 1.5
AUTO_EXPOSURE_RESET_MS = 1500  # ImSwitch returns to manual exposure after this time
FRAME_RESIZE = 0.25            # the frames are compared at this share of their size
# One LED must change the image by max(MIN_CHANGE, noise * NOISE_FACTOR) grey values.
# The noise is the largest difference between NOISE_SAMPLES dark frames.
MIN_CHANGE = 0.5
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


def every_led_on():
    client.get_json("/LEDMatrixController/setAllLED", {
        "intensity_r": MATRIX_INTENSITY, "intensity_g": MATRIX_INTENSITY,
        "intensity_b": MATRIX_INTENSITY})
    time.sleep(LIGHT_SETTLE_S)


def only_led_on(led):
    """Only ``led`` on: setLEDs switches every other LED off."""
    client.get_json("/LEDMatrixController/setLEDs", {
        "ledIds": str(led), "intensity_r": SINGLE_INTENSITY, "intensity_g": SINGLE_INTENSITY,
        "intensity_b": SINGLE_INTENSITY})
    time.sleep(LIGHT_SETTLE_S)


def central_led():
    """Chain index of the LED closest to the optical axis.

    The DPC controller knows every LED's illumination NA; the smallest NA is the most central LED.
    Without it, the LED in the middle of the setup's Nx x Ny grid is taken.
    The order of the LEDs in the chain is not known here, so that LED may lie off centre.
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
# An acquisition detector ("forAcquisition": true) gives frames only while its live view runs.
# The observation camera ("forAcquisition": false) gives them at any time.

class LeftOut(Exception):
    """A camera that cannot give frames now; the test leaves it out and names it in the report."""


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
                               params={"slot_id": "1", "camera_name": "hil_ledmatrix_single"})
        if response.status_code == 400:
            raise LeftOut(f"observation camera not connected: {response.text}")
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
        raise LeftOut("exposure too long for live view")
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


def sees_one_led(detector, led):
    """Every LED, then dark, then the one LED on and off, in front of one camera.

    The exposure is set on every LED, the brightest scene, so that no frame saturates;
    all frames of the camera share it.
    The observation camera has no live view for the auto exposure to act on.
    Returns what the camera did not see, as lines; empty when it saw everything.
    """
    observation = is_observation_camera(detector)
    with contextlib.nullcontext() if observation else live_view(detector):
        every_led_on()
        if not observation:
            auto_expose(detector)
        every = take_image(detector)
        all_lights_off()
        dark, noise_floor, required = dark_baseline(detector)
        only_led_on(led)
        single = take_image(detector)
        all_lights_off()
        off = take_image(detector)
        settings = exposure_and_gain(detector)
    attach(f"{detector}: dark", dark)
    attach(f"{detector}: LED {led} alone", single)
    attach(f"{detector}: off again", off)
    attach(f"{detector}: every LED", every)
    on_change, off_change = image_difference(dark, single), image_difference(dark, off)
    print(f"\n{detector}: LED {led} on changed {on_change:.2f}, "
          f"off is {off_change:.2f} from dark, "
          f"noise floor {noise_floor:.2f}, required {required:.2f}; brightness dark "
          f"{brightness(dark):.2f}, single {brightness(single):.2f}, "
          f"all {brightness(every):.2f}; {settings}")
    missed = []
    if on_change < required:
        missed.append(f"{detector}: LED {led} alone changed the image only by {on_change:.2f}, "
                      f"need >= {required:.2f} (noise floor {noise_floor:.2f}; {settings}) "
                      f"- setLEDs may not have lit it")
    # Off is judged against on, not against the noise:
    # the observation camera's own exposure drifts after a bright light.
    if off_change >= on_change / 2:
        missed.append(f"{detector}: off is still {off_change:.2f} from dark, "
                      f"need < {on_change / 2:.2f} (half of what on changed)")
    every_change = image_difference(dark, every)
    if every_change <= on_change:
        missed.append(f"{detector}: every LED changed the image by {every_change:.2f}, not more "
                      f"than LED {led} alone ({on_change:.2f}) - setLEDs may have lit more "
                      f"than one LED, or the camera saturates")
    return missed


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def led_matrix():
    if not client.setup()["LEDMatrixs"]:
        pytest.skip("the setup has no LED matrix")
    if client.get("/LEDMatrixController/setLEDs", params={"ledIds": ""}).status_code == 404:
        pytest.skip("this ImSwitch has no LEDMatrixController/setLEDs")


@pytest.fixture
def dark():
    """Every light off before and after the test."""
    all_lights_off()
    yield
    all_lights_off()


# --- tests ------------------------------------------------------------------------------------

def test_every_camera_sees_one_led_switch_on_and_off(dark):
    found = cameras()
    if not found:
        pytest.skip("the setup has no camera that gives real frames")
    led, why = central_led()
    note(f"LED {led}: {why}")
    missed, measured = [], 0
    # Every camera is measured before the verdict, so the report has the photos of all of them.
    for detector in found:
        try:
            missed += sees_one_led(detector, led)
            measured += 1
        except LeftOut as reason:
            note(f"{detector} left out: {reason}")
            all_lights_off()
    assert not missed, "\n".join(missed)
    if not measured:
        pytest.skip("no camera of the setup could give frames")
