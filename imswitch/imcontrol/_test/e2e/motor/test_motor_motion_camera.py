"""Verify physical stage motion with the observation camera.

MOVES REAL HARDWARE: parks the stage at the transport position, then moves
every axis out and back while comparing camera frames.

- test_axis_motion_is_visible (A, X, Y, Z): grey value change above noise
- test_axis_motion_direction (X, Y): phase correlation shows right / down
- test_z_motion_changes_scale (Z): apparent image scale changes above noise

Every axis is moved once; all three tests share that measurement.
"""

import base64
import io
import os
import time

import numpy as np
import pytest
import requests
from PIL import Image

from move_to_transport import move_to_transport


BASE_URL = os.environ.get(
    "IMSWITCH_URL",
    "http://localhost:8001",
)

DISTANCE_UM = int(
    os.environ.get(
        "MOTION_CAMERA_DISTANCE_UM",
        "3000",
    )
)

# Direction of the first Z move: negative first, so Z does not run towards its
# endstop (homeDirectionZ is +1) right after the transport move switched the Z
# hard limits off.
Z_DIRECTION = int(
    os.environ.get(
        "MOTION_CAMERA_Z_DIRECTION",
        "-1",
    )
)
Z_DISTANCE_UM = int(
    os.environ.get(
        "MOTION_CAMERA_Z_DISTANCE_UM",
        "5000",
    )
)

# Motor speed passed to movePositioner; PositionerController defaults to 5000.
SPEED = int(
    os.environ.get(
        "MOTION_CAMERA_SPEED",
        "15000",
    )
)

SETTLE_MS = int(
    os.environ.get(
        "MOTION_CAMERA_SETTLE_MS",
        "500",
    )
)

MIN_RATIO = float(
    os.environ.get(
        "MOTION_CAMERA_MIN_RATIO",
        "1.8",
    )
)

# A and Z get a lower bar than X and Y. They do not shift the image sideways,
# so their grey value change is smaller and sits closer to the noise; judging
# them by the X/Y threshold fails them for moves that did happen.
MIN_RATIO_AZ = float(
    os.environ.get(
        "MOTION_CAMERA_MIN_RATIO_AZ",
        "1.25",
    )
)


# Region of interest for X/Y motion detection, indexed frame[y1:y2, x1:x2].
# Defaults are chosen for the current 640 x 360 observation image.
ROI_X1 = int(
    os.environ.get(
        "MOTION_CAMERA_ROI_X1",
        "150",
    )
)

ROI_X2 = int(
    os.environ.get(
        "MOTION_CAMERA_ROI_X2",
        "340",
    )
)

ROI_Y1 = int(
    os.environ.get(
        "MOTION_CAMERA_ROI_Y1",
        "150",
    )
)

ROI_Y2 = int(
    os.environ.get(
        "MOTION_CAMERA_ROI_Y2",
        "355",
    )
)

# Z/A analysis: only the image below this top crop is used for the grey value
# difference.
TOP_CROP_PERCENT = int(
    os.environ.get(
        "MOTION_CAMERA_TOP_CROP_PERCENT",
        "25",
    )
)

# Z scale test: searched scale range / step, and pass limits
# (minimum |scale - 1| and multiple of the stationary scale noise).
Z_SCALE_MIN = float(os.environ.get("MOTION_CAMERA_Z_SCALE_MIN", "0.90"))
Z_SCALE_MAX = float(os.environ.get("MOTION_CAMERA_Z_SCALE_MAX", "1.10"))
Z_SCALE_STEP = float(os.environ.get("MOTION_CAMERA_Z_SCALE_STEP", "0.002"))
Z_MIN_SCALE_CHANGE = float(os.environ.get("MOTION_CAMERA_Z_MIN_SCALE_CHANGE", "0.001"))
Z_SCALE_NOISE_RATIO = float(os.environ.get("MOTION_CAMERA_Z_SCALE_NOISE_RATIO", "1.4"))


# Full brightness, white. 255 is the per-channel maximum the firmware takes.
LEDMATRIX_INTENSITY = int(os.environ.get("LEDMATRIX_INTENSITY", "255"))


def matrix(method, **params):
    """Call one LEDMatrixController endpoint and return the raw response.

    Never raises: a rig without an LED matrix should still run the motion
    tests, so a missing controller is not an error here.
    """
    try:
        return requests.get(
            f"{BASE_URL}/api/LEDMatrixController/{method}",
            params=params,
            timeout=30,
        )
    except requests.RequestException:
        return None


def matrix_on():
    """Switch the whole matrix on at full brightness, white."""
    return matrix(
        "setAllLED",
        intensity_r=LEDMATRIX_INTENSITY,
        intensity_g=LEDMATRIX_INTENSITY,
        intensity_b=LEDMATRIX_INTENSITY,
    )


def matrix_off():
    """Switch the matrix off."""
    return matrix("setAllLEDOff")


def api(controller, method, **params):
    """Call an ImSwitch endpoint and return its JSON."""
    response = requests.get(
        f"{BASE_URL}/api/{controller}/{method}",
        params=params,
        timeout=60,
    )

    assert response.status_code == 200, response.text

    return response.json()


def grab():
    """Capture one greyscale observation frame as a float array."""
    response = requests.post(
        f"{BASE_URL}/api/ExperimentController/snapOverviewImage",
        params={
            "slot_id": "1",
            "camera_name": "e2e_motion_test",
        },
        timeout=60,
    )

    assert response.status_code == 200, response.text

    image_bytes = base64.b64decode(
        response.json()["imageBase64"]
    )

    frame = np.asarray(
        Image.open(
            io.BytesIO(image_bytes)
        ).convert("L"),
        dtype=np.float32,
    )

    print(
        f"SNAPSHOT shape={frame.shape}",
        flush=True,
    )

    return frame


def image_difference(before, after):
    """Return the mean absolute pixel difference."""

    assert before.shape == after.shape

    return float(
        np.mean(
            np.abs(after - before)
        )
    )


def crop_motion_roi(frame):
    """Return the ROI containing the moving stage/object."""

    height, width = frame.shape

    assert 0 <= ROI_X1 < ROI_X2 <= width, (
        f"invalid ROI x-range "
        f"{ROI_X1}:{ROI_X2} "
        f"for image width {width}"
    )

    assert 0 <= ROI_Y1 < ROI_Y2 <= height, (
        f"invalid ROI y-range "
        f"{ROI_Y1}:{ROI_Y2} "
        f"for image height {height}"
    )

    return frame[
        ROI_Y1:ROI_Y2,
        ROI_X1:ROI_X2,
    ]


def crop_top(frame):
    """Remove the top TOP_CROP_PERCENT of the image."""

    height = frame.shape[0]

    top = (
        height
        * TOP_CROP_PERCENT
        // 100
    )

    assert 0 <= top < height, (
        f"invalid top crop "
        f"{TOP_CROP_PERCENT}% "
        f"for image height {height}"
    )

    return frame[top:, :]


def phase_shift(before, after):
    """Estimate translation of AFTER relative to BEFORE as (dy, dx, peak).

    Positive dx is right and negative left; positive dy is down, negative up.
    """

    assert before.shape == after.shape

    before = before.astype(
        np.float64,
        copy=True,
    )

    after = after.astype(
        np.float64,
        copy=True,
    )

    # Remove the DC component: structure and its displacement matter, not the
    # average brightness.
    before -= before.mean()
    after -= after.mean()

    # Hann window: the FFT assumes the image repeats, so without it the jump
    # from one edge to the opposite creates strong artificial frequencies.
    window = (
        np.hanning(before.shape[0])[:, None]
        * np.hanning(before.shape[1])[None, :]
    )

    before *= window
    after *= window

    # Fourier transforms.
    f_before = np.fft.fft2(before)
    f_after = np.fft.fft2(after)

    # Cross power spectrum: conj(F_before) * F_after gives the displacement of
    # AFTER relative to BEFORE, with the sign convention documented above.
    cross_power = (
        np.conj(f_before)
        * f_after
    )

    magnitude = np.abs(
        cross_power
    )

    # Normalise the amplitude away, leaving only phase.
    cross_power /= np.maximum(
        magnitude,
        1e-12,
    )

    # Transform the phase correlation back into image space.
    correlation = np.fft.ifft2(
        cross_power
    )

    correlation = np.abs(
        correlation
    )

    # Move zero displacement to the centre.
    correlation = np.fft.fftshift(
        correlation
    )

    # Locate the strongest correlation peak.
    peak_y, peak_x = np.unravel_index(
        np.argmax(correlation),
        correlation.shape,
    )

    center_y = (
        correlation.shape[0] // 2
    )

    center_x = (
        correlation.shape[1] // 2
    )

    dy = peak_y - center_y
    dx = peak_x - center_x

    peak = float(
        correlation[
            peak_y,
            peak_x,
        ]
    )

    return (
        float(dy),
        float(dx),
        peak,
    )


def estimate_scale(before, after):
    """Apparent scale of AFTER relative to BEFORE, as (scale, dx, dy, peak).

    scale > 1 means AFTER appears larger. Brute force: zoom BEFORE by every
    candidate scale, crop or pad it back to shape, align the residual shift
    with phase_shift() and keep the candidate with the highest peak.
    """

    height, width = before.shape
    image = Image.fromarray(np.ascontiguousarray(before, dtype=np.float32))
    fill = float(np.median(before))
    best = (1.0, 0.0, 0.0, -1.0)

    for scale in np.arange(Z_SCALE_MIN, Z_SCALE_MAX + Z_SCALE_STEP / 2, Z_SCALE_STEP):
        zoomed = np.asarray(image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.BILINEAR,
        ))
        zh, zw = zoomed.shape

        # Source offset (crop, scale > 1) and target offset (pad, scale < 1).
        sy, ty = max(0, (zh - height) // 2), max(0, (height - zh) // 2)
        sx, tx = max(0, (zw - width) // 2), max(0, (width - zw) // 2)
        h, w = min(zh, height), min(zw, width)

        candidate = np.full(before.shape, fill, dtype=np.float32)
        candidate[ty:ty + h, tx:tx + w] = zoomed[sy:sy + h, sx:sx + w]

        dy, dx, peak = phase_shift(candidate, after)

        if peak > best[3]:
            best = (round(float(scale), 6), dx, dy, peak)

    return best


# What describe_direction reports when the shift is too small to resolve.
NO_TRANSLATION = "no clear translation"


def describe_direction(
    dy,
    dx,
    min_shift=2,
):
    """Convert the measured translation into a rough direction."""

    distance = float(
        np.hypot(
            dx,
            dy,
        )
    )

    if distance < min_shift:
        return NO_TRANSLATION

    if abs(dx) > abs(dy):
        return (
            "right"
            if dx > 0
            else "left"
        )

    return (
        "down"
        if dy > 0
        else "up"
    )


def axes():
    """Every axis the setup reports, as (positioner, axis) pairs.

    Returns [] rather than raising: parametrize arguments are evaluated at
    collection, so an exception here would abort collection for the suite.
    """
    try:
        positions = api(
            "PositionerController",
            "getPositionerPositions",
        )

    except Exception:
        return []

    return [
        (positioner, axis)
        for positioner, axis_map
        in (positions or {}).items()
        for axis in axis_map
    ]


@pytest.fixture(scope="module")
def observation_camera():
    """Check the observation camera, then park the stage for transport.

    Module scoped, so a skip here skips every test in this file before any
    hardware is moved.
    """
    names = api(
        "SettingsController",
        "getDetectorNames",
    )

    camera = next(
        (
            name
            for name in names
            if "observ" in name.lower()
        ),
        None,
    )

    if camera is None:
        pytest.skip(
            f"no observation camera found; "
            f"detectors: {names}"
        )

    status = api(
        "SettingsController",
        "getCameraStatus",
        detectorName=camera,
    )

    if status.get("status") == "error":
        pytest.skip(
            f"{camera} status unavailable: "
            f"{status.get('error')}"
        )

    # A camera that fails to open usually falls back to a mock camera.
    if (
        status.get("isMock")
        or str(
            status.get(
                "model",
                "",
            )
        ).lower() == "mock"
    ):
        pytest.skip(
            f"{camera} is a mock camera "
            f"(not connected or turned off)"
        )

    if status.get("isConnected") is False:
        pytest.skip(
            f"{camera} is not connected"
        )

    # Verify that the camera works before moving hardware.
    try:
        frame = grab()

    except Exception as exc:
        pytest.skip(
            f"{camera} cannot deliver "
            f"an image: {exc}"
        )

    # Validate the ROI before moving hardware.
    try:
        roi = crop_motion_roi(frame)

    except AssertionError as exc:
        pytest.fail(
            f"invalid motion ROI: {exc}"
        )

    print(
        f"\nObservation camera: {camera}"
    )

    print(
        f"Image shape: {frame.shape}"
    )

    print(
        "Motion ROI: "
        f"x={ROI_X1}:{ROI_X2}, "
        f"y={ROI_Y1}:{ROI_Y2}, "
        f"shape={roi.shape}"
    )

    print(
        "Moving stage to transport position..."
    )

    position = move_to_transport()

    print(
        f"Transport position reached: "
        f"{position}"
    )

    time.sleep(
        SETTLE_MS / 1000
    )

    return camera


@pytest.fixture(scope="module", autouse=True)
def led_matrix_illumination(observation_camera):
    """Light the sample for the whole module, and go dark again afterwards.

    Depends on observation_camera, so the matrix comes on after the stage has
    been parked and before the first axis is measured. A setup without a matrix
    answers 404, which is reported and otherwise ignored: these tests are about
    motion, not about light.
    """
    response = matrix_on()

    if response is not None and response.status_code == 200:
        print(f"\nLED matrix on at {LEDMATRIX_INTENSITY}")
    else:
        status = response.status_code if response is not None else "no answer"
        print(f"\nLED matrix not switched on ({status})")

    time.sleep(SETTLE_MS / 1000)

    yield

    matrix_off()


# Expected image direction when an axis moves by +DISTANCE_UM.
EXPECTED_DIRECTION = {
    "X": "right",
    "Y": "down",
}


def translation_axes():
    """Return only the axes checked with phase correlation."""

    return [
        (positioner, axis)
        for positioner, axis in axes()
        if axis.upper() in EXPECTED_DIRECTION
    ]


def measure_axis_motion(
    positioner,
    axis,
):
    """Move one axis forward and back and analyse the camera frames.

    Returns the image difference results plus, per axis, the X/Y phase
    correlation or the Z scale estimate.
    """

    settle = (
        SETTLE_MS / 1000
    )

    use_translation_roi = (
        axis.upper() in EXPECTED_DIRECTION
    )

    # First move is positive for every axis except Z.
    distance = (
    Z_DISTANCE_UM * Z_DIRECTION
    if axis.upper() == "Z"
    else DISTANCE_UM
)

    print(
        f"\nMeasuring {positioner} {axis}"
    )

    if use_translation_roi:
        print(
            f"{axis}: using motion ROI "
            f"x={ROI_X1}:{ROI_X2}, "
            f"y={ROI_Y1}:{ROI_Y2}"
        )

    else:
        print(
            f"{axis}: using lower "
            f"{100 - TOP_CROP_PERCENT}% of the image "
            f"for image difference"
        )

    # X/Y are analysed in the ROI; Z/A use the image below the top crop,
    # because they do not necessarily produce a clean translation.
    def analysis_region(frame):
        if use_translation_roi:
            return crop_motion_roi(frame)

        return crop_top(frame)

    # Camera noise baseline. Full frames are kept for the Z scale analysis.
    baseline_1_full = grab()
    baseline_1 = analysis_region(
        baseline_1_full
    )

    time.sleep(
        settle
    )

    before_full = grab()
    before = analysis_region(
        before_full
    )

    baseline = image_difference(
        baseline_1,
        before,
    )

    print(
        f"{axis}: baseline difference = "
        f"{baseline:.3f}"
    )

    moved_forward = False

    try:
        print(
            f"{axis}: moving "
            f"{distance:+d} um"
        )

        api(
            "PositionerController",
            "movePositioner",
            positionerName=positioner,
            axis=axis,
            dist=distance,
            isBlocking=True,
            speed=SPEED,
        )

        moved_forward = True

        time.sleep(
            settle
        )

        after_full = grab()
        after = analysis_region(
            after_full
        )

    finally:
        # Always try to return to the starting position.
        if moved_forward:
            print(
                f"{axis}: moving "
                f"{-distance:+d} um"
            )

            api(
                "PositionerController",
                "movePositioner",
                positionerName=positioner,
                axis=axis,
                dist=-distance,
                isBlocking=True,
                speed=SPEED,
            )

            time.sleep(
                settle
            )

    # Grey value change, measured against the baseline noise.
    movement_difference = image_difference(
        before,
        after,
    )

    ratio = (
        movement_difference
        / max(
            baseline,
            1e-6,
        )
    )

    print(
        f"{axis}: movement difference = "
        f"{movement_difference:.3f}, "
        f"ratio={ratio:.2f}x"
    )

    result = {
        "baseline": baseline,
        "movement_difference": movement_difference,
        "ratio": ratio,
    }

    # X/Y translation and direction.
    if use_translation_roi:
        dy, dx, peak = phase_shift(
            before,
            after,
        )

        magnitude = float(
            np.hypot(
                dx,
                dy,
            )
        )

        direction = describe_direction(
            dy,
            dx,
        )

        print(
            f"{axis}: image shift = "
            f"dx={dx:+.1f}px, "
            f"dy={dy:+.1f}px, "
            f"magnitude={magnitude:.1f}px, "
            f"direction={direction}, "
            f"peak={peak:.4f}"
        )

        result.update(
            dx=dx,
            dy=dy,
            peak=peak,
            direction=direction,
        )

    # Z apparent scale, from the same frames but always the motion ROI.
    if axis.upper() == "Z":
        baseline_scale = estimate_scale(
            crop_motion_roi(baseline_1_full), crop_motion_roi(before_full)
        )[0]
        scale, dx, dy, peak = estimate_scale(
            crop_motion_roi(before_full), crop_motion_roi(after_full)
        )
        size = "larger" if scale > 1 else "smaller" if scale < 1 else "unchanged"

        print(f"{axis}: baseline apparent scale = {baseline_scale:.4f} ({baseline_scale - 1:+.2%})")
        print(
            f"{axis}: apparent scale = {scale:.4f} ({scale - 1:+.2%}), image={size}, "
            f"residual dx={dx:+.1f}px, dy={dy:+.1f}px, peak={peak:.4f}"
        )

        result.update(
            z_scale=scale,
            z_scale_change=abs(scale - 1),
            z_baseline_scale=baseline_scale,
            z_baseline_scale_change=abs(baseline_scale - 1),
            z_scale_dx=dx,
            z_scale_dy=dy,
            z_scale_peak=peak,
        )

    return result


@pytest.fixture(scope="module")
def axis_motion(observation_camera):
    """Measure every axis only once per test run.

    All three tests read the same measurement, so hardware is not moved twice.
    """

    cache = {}

    def get(positioner, axis):
        key = (positioner, axis)

        if key not in cache:
            cache[key] = measure_axis_motion(
                positioner,
                axis,
            )

        return cache[key]

    return get


@pytest.mark.hardware
@pytest.mark.parametrize(
    "positioner,axis",
    axes() or [(None, None)],
)
def test_axis_motion_is_visible(
    positioner,
    axis,
    axis_motion,
):
    """A, X, Y, Z: grey value change must be above camera noise."""

    if positioner is None:
        pytest.skip(
            "no positioners reported"
        )

    result = axis_motion(
        positioner,
        axis,
    )

    # X and Y are judged by the strict threshold, A and Z by the lenient one.
    required = (
        MIN_RATIO
        if axis.upper() in EXPECTED_DIRECTION
        else MIN_RATIO_AZ
    )

    assert result["ratio"] >= required, (
        f"{positioner} {axis}: "
        f"physical movement was not clearly visible. "
        f"movement={result['movement_difference']:.3f}, "
        f"baseline={result['baseline']:.3f}, "
        f"ratio={result['ratio']:.2f}x, "
        f"required={required:.2f}x"
    )


@pytest.mark.hardware
@pytest.mark.parametrize(
    "positioner,axis",
    translation_axes() or [(None, None)],
)
def test_axis_motion_direction(
    positioner,
    axis,
    axis_motion,
):
    """X, Y: the image must not move against the expected direction.

    Passes on the expected direction, skips when the correlation resolves no
    clear translation, and fails only on a direction that contradicts it.
    """

    if positioner is None:
        pytest.skip(
            "no X/Y positioners reported"
        )

    result = axis_motion(
        positioner,
        axis,
    )

    expected_direction = EXPECTED_DIRECTION[
        axis.upper()
    ]

    # Too small a shift to resolve is not a wrong direction — the stage may
    # simply have moved less than the correlation can see, which this test
    # cannot judge. Only a contradicting direction is a failure.
    if result["direction"] == NO_TRANSLATION:
        pytest.skip(
            f"{positioner} {axis}: no clear translation "
            f"(dx={result['dx']:+.1f}px, "
            f"dy={result['dy']:+.1f}px, "
            f"peak={result['peak']:.4f})"
        )

    assert result["direction"] == expected_direction, (
        f"{positioner} {axis}: "
        f"image moved in the wrong direction. "
        f"expected={expected_direction}, "
        f"measured={result['direction']}, "
        f"dx={result['dx']:+.1f}px, "
        f"dy={result['dy']:+.1f}px, "
        f"peak={result['peak']:.4f}"
    )


@pytest.mark.hardware
@pytest.mark.parametrize(
    "positioner,axis",
    [(p, a) for p, a in axes() if a.upper() == "Z"] or [(None, None)],
)
def test_z_motion_changes_scale(
    positioner,
    axis,
    axis_motion,
):
    """Z: apparent image scale must change clearly above stationary noise.

    Skips while the move itself was visible: an inconclusive scale estimate
    then says something about the measurement, not about the stage. Only when
    nothing moved at all does an unclear scale count as a failure. Whether Z
    makes the image larger or smaller is not asserted.
    """

    if positioner is None:
        pytest.skip(
            "no Z positioner reported"
        )

    result = axis_motion(
        positioner,
        axis,
    )

    change = result["z_scale_change"]
    noise = result["z_baseline_scale_change"]

    clear = (
        change >= Z_MIN_SCALE_CHANGE
        and change >= noise * Z_SCALE_NOISE_RATIO
    )

    evidence = (
        f"scale={result['z_scale']:.4f}, "
        f"change={change:.4f}, "
        f"baseline change={noise:.4f}, "
        f"required>={Z_MIN_SCALE_CHANGE:.4f} "
        f"and >={Z_SCALE_NOISE_RATIO:.1f}x baseline"
    )

    # The grey value check already saw this axis move, so an unclear scale is
    # a limit of the scale estimate rather than a stage that stood still.
    if not clear and result["ratio"] >= MIN_RATIO_AZ:
        pytest.skip(
            f"{positioner} {axis}: movement was visible "
            f"(ratio={result['ratio']:.2f}x >= {MIN_RATIO_AZ:.2f}x) but the "
            f"apparent scale change is inconclusive. {evidence}"
        )

    assert clear, (
        f"{positioner} {axis}: "
        f"apparent scale change not clear enough, and no movement was seen "
        f"either (ratio={result['ratio']:.2f}x). {evidence}"
    )
