"""Stage motion as the observation camera sees it. MOVES REAL HARDWARE and SWITCHES LIGHT.

The stage parks at the transport position (ImSwitch switches the Z hard limits off there, and
the turret goes to slot 0) and the LED matrix lights the sample. Then every axis moves out and
back once, and three tests judge that one measurement:

- test_axis_motion_is_visible (every axis): the image changed more than its noise
- test_axis_motion_direction (X, Y): +X moves the image right, +Y moves it down
- test_z_motion_changes_scale (Z): the image got larger or smaller
"""
import base64
import functools
import io
import time

import numpy as np
import pytest
from PIL import Image

import hil_client as client
from hil_report import attach

# Distances are in µm, times in seconds.
DISTANCE_UM = 3000
Z_DISTANCE_UM = 5000
Z_DIRECTION = -1               # away from Z home: at the transport position the limits are off
SPEED = 15000
SETTLE_S = 0.5                 # wait after a move, before the frame
TRANSPORT_SPEED = 20000        # above 30000 the stage loses steps
TRANSPORT_TIMEOUT = 120
TURRET_TIMEOUT = 120
MATRIX_INTENSITY = 255         # the firmware's maximum per colour
MATRIX_SETTLE_S = 3.5          # wait after switching the matrix on, before the first frame

EXPECTED_DIRECTION = {"X": "right", "Y": "down"}  # of the image, when the axis moves by +distance
MIN_RATIO = 1.8                # image change over noise that counts as movement, X and Y
MIN_RATIO_AZ = 1.25            # the same for A and Z, which change the image less
ROI = (150, 355, 150, 340)     # X/Y are judged in frame[y1:y2, x1:x2] of the 640x360 image
TOP_CROP_PERCENT = 25          # A/Z are judged below this top share of the image
Z_SCALE_MIN = 0.90             # the image scales that are tried to find the Z zoom
Z_SCALE_MAX = 1.10
Z_SCALE_STEP = 0.002
Z_MIN_SCALE_CHANGE = 0.001
Z_SCALE_NOISE_RATIO = 1.4

SNAP = {"slot_id": "1", "camera_name": "hil_motion"}  # camera_name: folder ImSwitch saves to


# --- stage ------------------------------------------------------------------------------------

def axes():
    """Every (positioner, axis) pair of the setup, e.g. ("ESP32Stage", "X")."""
    return [(name, axis) for name, positioner in client.setup()["positioners"].items()
            if positioner["forPositioning"] for axis in positioner["axes"]]


def xy_axes():
    return [pair for pair in axes() if pair[1].upper() in EXPECTED_DIRECTION]


def z_axes():
    return [pair for pair in axes() if pair[1].upper() == "Z"]


def move(positioner, axis, distance):
    """Move one axis by ``distance`` µm and wait until it has arrived and settled."""
    client.get_json("/PositionerController/movePositioner", timeout=60, params={
        "positionerName": positioner, "axis": axis, "dist": distance,
        "isBlocking": True, "speed": SPEED})
    time.sleep(SETTLE_S)


def wait_until_stopped():
    """The positions once two reads agree; some endpoints answer before the stage has arrived."""
    previous = client.get_json("/PositionerController/getPositionerPositions")
    deadline = time.monotonic() + TRANSPORT_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(0.5)
        current = client.get_json("/PositionerController/getPositionerPositions")
        if current == previous:
            break
        previous = current
    return previous


def park_at_transport():
    """Move to the transport position and turn the turret to slot 0; returns the positions."""
    client.get_json("/PositionerController/moveToTransportPosition", timeout=TRANSPORT_TIMEOUT,
                    params={"speed": TRANSPORT_SPEED, "isBlocking": True})
    wait_until_stopped()
    if client.setup()["objective"]:
        status = client.get_json("/ObjectiveController/getstatus")
        if status.get("isActive") and status.get("hasMotor") and status["slotConfigured"][0]:
            # skipZ: the turret must not apply the slot's Z offset to the parked stage.
            started = client.get_json("/ObjectiveController/moveToObjective",
                                      {"slot": 0, "skipZ": True})
            assert started.get("accepted"), f"the turret refused slot 0: {started}"

            def turned():
                # moveToObjective only starts the move; the status tells when it is done.
                now = client.get_json("/ObjectiveController/getstatus")
                return (not now.get("isMovingObjective") and
                        now.get("lastCompletedObjectiveRequestId") == started["requestId"])

            assert client.wait_until(turned, TURRET_TIMEOUT), (
                f"the turret did not reach slot 0 in {TURRET_TIMEOUT}s")
    return wait_until_stopped()


# --- observation camera -----------------------------------------------------------------------

def frame():
    """One full-size greyscale frame of the observation camera, as a float array."""
    body = client.post_json("/ExperimentController/snapOverviewImage", params=SNAP, timeout=60)
    image = Image.open(io.BytesIO(base64.b64decode(body["imageBase64"]))).convert("L")
    return np.asarray(image, dtype=np.float32)


# --- image maths: frames are greyscale float arrays -------------------------------------------

def difference(before, after):
    """Mean absolute grey-value difference."""
    assert before.shape == after.shape
    return float(np.mean(np.abs(after - before)))


def roi(image):
    """The part of the frame in which X/Y motion is judged (ROI)."""
    y1, y2, x1, x2 = ROI
    height, width = image.shape
    assert 0 <= x1 < x2 <= width, f"invalid ROI x-range {x1}:{x2} for image width {width}"
    assert 0 <= y1 < y2 <= height, f"invalid ROI y-range {y1}:{y2} for image height {height}"
    return image[y1:y2, x1:x2]


def below_top(image):
    """The frame without its top TOP_CROP_PERCENT, where A/Z motion is judged."""
    return image[image.shape[0] * TOP_CROP_PERCENT // 100:, :]


def shift(before, after):
    """How far AFTER is moved against BEFORE: (dx, dy, peak) in pixels, +dx right, +dy down.

    Phase correlation: the peak of the correlation of the two images lies at their shift, and
    its height says how clear the match is.
    """
    assert before.shape == after.shape
    before, after = before.astype(np.float64), after.astype(np.float64)
    # Fade the edges out, or the FFT takes the jump from one edge to the other for structure.
    window = np.hanning(before.shape[0])[:, None] * np.hanning(before.shape[1])[None, :]
    f_before = np.fft.fft2((before - before.mean()) * window)
    f_after = np.fft.fft2((after - after.mean()) * window)
    cross_power = np.conj(f_before) * f_after
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.fftshift(np.abs(np.fft.ifft2(cross_power)))
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy = peak_y - correlation.shape[0] // 2
    dx = peak_x - correlation.shape[1] // 2
    return float(dx), float(dy), float(correlation[peak_y, peak_x])


def direction(dx, dy, min_shift=2):
    """ "right", "left", "down" or "up"; None if the shift is under ``min_shift`` pixels."""
    if np.hypot(dx, dy) < min_shift:
        return None
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def scale(before, after):
    """The zoom of AFTER against BEFORE; above 1 means the image got larger.

    Tries every scale from Z_SCALE_MIN to Z_SCALE_MAX: zoom BEFORE by it, cut or pad it back
    to the frame size, and keep the scale whose result matches AFTER best.
    """
    height, width = before.shape
    image = Image.fromarray(np.ascontiguousarray(before, dtype=np.float32))
    best_scale, best_peak = 1.0, -1.0
    for candidate in np.arange(Z_SCALE_MIN, Z_SCALE_MAX + Z_SCALE_STEP / 2, Z_SCALE_STEP):
        zoomed = np.asarray(image.resize((max(1, round(width * candidate)),
                                          max(1, round(height * candidate))), Image.BILINEAR))
        zoomed_height, zoomed_width = zoomed.shape
        # Offsets in the zoomed image (when it is larger) and in the frame (when it is smaller).
        from_y, to_y = max(0, (zoomed_height - height) // 2), max(0, (height - zoomed_height) // 2)
        from_x, to_x = max(0, (zoomed_width - width) // 2), max(0, (width - zoomed_width) // 2)
        h, w = min(zoomed_height, height), min(zoomed_width, width)
        resized = np.full(before.shape, float(np.median(before)), dtype=np.float32)
        resized[to_y:to_y + h, to_x:to_x + w] = zoomed[from_y:from_y + h, from_x:from_x + w]
        peak = shift(resized, after)[2]
        if peak > best_peak:
            best_scale, best_peak = round(float(candidate), 6), peak
    return best_scale


# --- the measurement and its tests ------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def parked_and_lit():
    """Check the camera before anything moves, then park the stage and switch the matrix on."""
    setup = client.setup()
    cameras = [name for name, detector in setup["detectors"].items()
               if not detector["forAcquisition"]]
    if not cameras:
        pytest.skip("the setup has no observation camera")
    status = client.get_json("/SettingsController/getCameraStatus", {"detectorName": cameras[0]})
    if "error" in status or status.get("status") == "error":
        pytest.skip(f"{cameras[0]}: getCameraStatus failed: {status.get('error')}")
    if status.get("isMock") or status.get("isConnected") is False:
        pytest.skip(f"{cameras[0]} is a mock camera or not connected: {status}")
    response = client.post("/ExperimentController/snapOverviewImage", params=SNAP, timeout=60)
    if response.status_code != 200:
        pytest.skip(f"{cameras[0]} cannot deliver an image: {response.text}")
    roi(frame())  # fails here if ROI does not fit the image

    print(f"\ntransport position reached: {park_at_transport()}")
    has_matrix = bool(setup["LEDMatrixs"])
    if has_matrix:
        client.get_json("/LEDMatrixController/setAllLED", {
            "intensity_r": MATRIX_INTENSITY, "intensity_g": MATRIX_INTENSITY,
            "intensity_b": MATRIX_INTENSITY})
        time.sleep(MATRIX_SETTLE_S)
    yield
    if has_matrix:
        client.get_json("/LEDMatrixController/setAllLEDOff")


@functools.lru_cache(maxsize=None)  # each axis moves once; the tests share the result
def measure(positioner, axis):
    """Move one axis out and back, and return what the frames before and after show."""
    is_xy = axis.upper() in EXPECTED_DIRECTION
    is_z = axis.upper() == "Z"
    distance = Z_DISTANCE_UM * Z_DIRECTION if is_z else DISTANCE_UM
    # X and Y shift the image, so they are judged in a region that stays on the sample.
    region = roi if is_xy else below_top

    still = frame()
    time.sleep(SETTLE_S)
    before = frame()
    move(positioner, axis, distance)
    try:
        after = frame()
    finally:
        move(positioner, axis, -distance)
    attach(f"{axis} before", before)
    attach(f"{axis} after {distance:+d} um", after)

    noise = difference(region(still), region(before))
    change = difference(region(before), region(after))
    result = {"noise": noise, "change": change, "ratio": change / max(noise, 1e-6)}
    if is_xy:
        dx, dy, peak = shift(region(before), region(after))
        result["direction"] = direction(dx, dy)
        result["shift"] = f"dx={dx:+.1f}px, dy={dy:+.1f}px, peak={peak:.4f}"
    if is_z:
        result["scale"] = scale(roi(before), roi(after))
        result["scale_change"] = abs(result["scale"] - 1)
        result["scale_noise"] = abs(scale(roi(still), roi(before)) - 1)
    print(f"\n{positioner} {axis} {distance:+d} um: {result}")
    return result


@client.parametrize_from_rig("positioner, axis", axes, "the setup has no positioner axes")
def test_axis_motion_is_visible(positioner, axis):
    result = measure(positioner, axis)
    required = MIN_RATIO if axis.upper() in EXPECTED_DIRECTION else MIN_RATIO_AZ
    assert result["ratio"] >= required, (
        f"{positioner} {axis}: movement was not clearly visible. change={result['change']:.3f}, "
        f"noise={result['noise']:.3f}, ratio={result['ratio']:.2f}x, required={required:.2f}x")


@client.parametrize_from_rig("positioner, axis", xy_axes, "the setup has no X/Y axes")
def test_axis_motion_direction(positioner, axis):
    """Only the wrong direction fails. A shift too small to measure skips."""
    result = measure(positioner, axis)
    if result["direction"] is None:
        pytest.skip(f"{positioner} {axis}: no clear translation ({result['shift']})")
    expected = EXPECTED_DIRECTION[axis.upper()]
    assert result["direction"] == expected, (
        f"{positioner} {axis}: image moved {result['direction']}, expected {expected} "
        f"({result['shift']})")


@client.parametrize_from_rig("positioner, axis", z_axes, "the setup has no Z axis")
def test_z_motion_changes_scale(positioner, axis):
    """Larger or smaller does not matter. An unclear scale skips as long as the move was seen."""
    result = measure(positioner, axis)
    clear = (result["scale_change"] >= Z_MIN_SCALE_CHANGE
             and result["scale_change"] >= result["scale_noise"] * Z_SCALE_NOISE_RATIO)
    evidence = (f"scale={result['scale']:.4f}, change={result['scale_change']:.4f}, "
                f"noise={result['scale_noise']:.4f}, ratio={result['ratio']:.2f}x")
    if not clear and result["ratio"] >= MIN_RATIO_AZ:
        pytest.skip(f"{positioner} {axis}: moved, but the scale change is unclear. {evidence}")
    assert clear, f"{positioner} {axis}: no scale change and no movement seen. {evidence}"
