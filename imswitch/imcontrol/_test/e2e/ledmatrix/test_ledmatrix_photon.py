"""Check that the LED matrix physically lights up the camera."""

import os
import time

import pytest
import requests
from PIL import ImageStat


# ImSwitch runs on :8001 without the caddy prefix inside the container; from
# outside the Pi it is http://<pi>:8000/imswitch, so set IMSWITCH_URL then.
BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")
DETECTOR = os.environ.get("IMSWITCH_DETECTOR")

INTENSITY = int(os.environ.get("LEDMATRIX_INTENSITY", "500"))

SETTLE = 0.5


def api(controller, method, http="GET", **params):
    """Call an ImSwitch endpoint and return its JSON."""
    send = requests.post if http == "POST" else requests.get

    response = send(
        f"{BASE_URL}/api/{controller}/{method}",
        params=params,
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.json()


def is_observation_camera(detector):
    """True for the observation camera, which has no acquisition path."""
    return "observ" in detector.lower()


def get_available_controllers():
    """Controllers of the active setup; the only way to spot the matrix."""
    response = requests.get(
        f"{BASE_URL}/api/getAvailableControllers",
        timeout=30,
    )

    assert response.status_code == 200, (
        f"getAvailableControllers -> "
        f"{response.status_code}: {response.text}"
    )

    return response.json()


def matrix(method, **params):
    """Call one LEDMatrixController endpoint and return the raw response.

    Never skips: matrix_off() also runs from teardown, where a pytest.skip
    would report the test a second time. The missing-controller skip happens
    once, in led_matrix_available.
    """
    return requests.get(
        f"{BASE_URL}/api/LEDMatrixController/{method}",
        params=params,
        timeout=30,
    )


def matrix_on():
    """Switch the matrix on at INTENSITY, white."""
    response = matrix(
        "setAllLED",
        intensity_r=INTENSITY,
        intensity_g=INTENSITY,
        intensity_b=INTENSITY,
    )

    assert response.status_code == 200, response.text
    time.sleep(SETTLE)


def matrix_off():
    """Switch the matrix off. Safe from teardown; 404 means no matrix."""
    try:
        response = matrix("setAllLEDOff")
    except requests.RequestException:
        return

    if response.status_code == 404:
        return

    assert response.status_code == 200, response.text
    time.sleep(SETTLE)


def lasers_off():
    """Switch off every laser/LED, so only the matrix can change the image."""
    try:
        names = api("LaserController", "getLaserNames")
    except requests.RequestException:
        return

    for name in names:
        api("LaserController", "setLaserValue", laserName=name, value=0)
        api("LaserController", "setLaserActive", laserName=name, active=False)


@pytest.fixture(scope="module", params=["acquisition", "observation"])
def detector_name(request):
    """Run every test twice: acquisition detector, then observation camera.

    The observation camera only exists on some rigs, so that parameter skips
    where the setup has none.
    """
    try:
        detectors = api("SettingsController", "getDetectorNames")
    except requests.RequestException as exc:
        pytest.skip(f"ImSwitch not reachable at {BASE_URL}: {exc}")

    if not detectors:
        pytest.skip("active setup has no detectors")

    if request.param == "observation":
        camera = next(
            (name for name in detectors if is_observation_camera(name)),
            None,
        )

        if camera is None:
            pytest.skip(f"no observation camera in this setup: {detectors}")

        return camera

    if DETECTOR:
        if DETECTOR not in detectors:
            pytest.skip(
                f"detector {DETECTOR!r} not found (available: {detectors})"
            )

        return DETECTOR

    # The observation camera has its own parameter, so it is not the fallback.
    acquisition = [
        name for name in detectors if not is_observation_camera(name)
    ]

    if not acquisition:
        pytest.skip("setup has no acquisition detector")

    return acquisition[0]


@pytest.fixture(scope="module")
def camera_status(detector_name):
    """Status of the detector under test; skips on an ImSwitch mock camera."""
    status = api(
        "SettingsController",
        "getCameraStatus",
        detectorName=detector_name,
    )

    if "error" in status:
        pytest.skip(
            f"{detector_name}: getCameraStatus failed: {status['error']}"
        )

    if status.get("isMock") or str(status.get("model", "")).lower() == "mock":
        pytest.skip(
            f"{detector_name}: ImSwitch served a mock camera "
            f"(model={status.get('model')!r}), no hardware attached"
        )

    return status


@pytest.fixture(scope="module")
def led_matrix_available():
    """Skip the module unless this setup actually has an LED matrix."""

    try:
        controllers = get_available_controllers()
    except requests.RequestException as exc:
        pytest.skip(f"ImSwitch not reachable at {BASE_URL}: {exc}")

    # The response is either a plain list or a dictionary.
    if isinstance(controllers, dict):
        controllers = controllers.get(
            "controllers",
            controllers.get("availableControllers", []),
        )

    if "LEDMatrixController" not in controllers:
        pytest.skip("LEDMatrixController not available in this setup")

        

@pytest.fixture(scope="module", autouse=True)
def camera_acquisition(led_matrix_available, detector_name, camera_status):
    """Keep the camera streaming for the duration of this module.

    LiveViewController owns the stream and reports real state; the older
    ViewController/setLiveViewActive has a stale handle at boot on these rigs.
    The observation camera is left alone: snapOverviewImage reads its latest
    frame directly, so it needs no stream of its own.
    """
    status = None

    if not is_observation_camera(detector_name):
        started = api(
            "LiveViewController",
            "startLiveView",
            http="POST",
            detectorName=detector_name,
        )
        status = started.get("status")

        # startLiveView answers 200 even when it declines, so the body decides.
        if status == "long_exposure":
            pytest.skip(f"{detector_name}: exposure too long for live view: {started}")

        # "already_running" is fine: an active stream is all this module needs.
        assert status in ("success", "already_running"), (
            f"{detector_name}: startLiveView did not start a stream: {started}"
        )

        assert api("LiveViewController", "getLiveViewActive") is True, (
            f"{detector_name}: getLiveViewActive is False right after "
            f"startLiveView returned {status!r}"
        )

        time.sleep(0.4)

    yield

    # The matrix must go dark even if stopping the stream fails.
    try:
        matrix_off()
    finally:
        # Only hand back what was taken: a stream that was already running
        # belongs to whoever started it.
        if status == "success":
            api(
                "LiveViewController",
                "stopLiveView",
                detectorName=detector_name,
            )


@pytest.fixture
def dark_rig():
    """Leave the rig dark before and after the test, matrix included."""
    lasers_off()
    matrix_off()

    yield

    lasers_off()
    matrix_off()


@pytest.mark.hardware
def test_led_matrix_is_visible_to_camera(
    dark_rig,
    detector_name,
    auto_exposure,
    measure_dark_baseline,
    take_image,
    image_difference,
):
    """The matrix must change the image by more than the camera noise floor.

    This compares two real frames, so unlike the state readbacks elsewhere in
    e2e/ it cannot pass without photons. Runs once per camera: the acquisition
    detector and, where the rig has one, the observation camera.
    """
    # Acquisition detector only: the pass runs once per session, and the
    # observation camera has no live stream for it to act on.
    if not is_observation_camera(detector_name):
        auto_exposure(detector_name)

    dark, noise_floor, required_change = measure_dark_baseline(
        detector_name
    )

    matrix_on()

    bright = take_image(detector_name)

    change = image_difference(dark, bright)

    print(
        f"\nLED matrix @ intensity {INTENSITY}: "
        f"dark_mean={ImageStat.Stat(dark).mean[0]:.2f} "
        f"bright_mean={ImageStat.Stat(bright).mean[0]:.2f} "
        f"noise_floor={noise_floor:.2f} "
        f"pixel_change={change:.2f} "
        f"required={required_change:.2f}"
    )

    assert bright.getextrema()[1] > 0, (
        "the matrix is on but every pixel is still 0; "
        "no light reached the sensor"
    )

    assert change >= required_change, (
        f"LED matrix: image changed only by {change:.2f}; "
        f"camera noise floor is {noise_floor:.2f}, "
        f"need >= {required_change:.2f}"
    )
