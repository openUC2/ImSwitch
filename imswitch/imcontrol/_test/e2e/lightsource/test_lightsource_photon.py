"""Check that every configured LED is visible to the camera."""

import os
import time

import pytest
import requests
from PIL import ImageStat


# ImSwitch runs on :8001 without the caddy prefix inside the container; from
# outside the Pi it is http://<pi>:8000/imswitch, so set IMSWITCH_URL then.
BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")
DETECTOR = os.environ.get("IMSWITCH_DETECTOR")
LASER_VALUE = int(os.environ.get("UC2_LASER_VALUE", "1000"))

SETTLE = 0.1


def api(controller, method, http="GET", **params):
    """Call an ImSwitch endpoint and return its JSON.

    A 200 does not always mean the call did what was asked: LiveViewController
    reports refusals in the body, so callers that care check the status field.
    """
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


def get_lights():
    """Names of all configured light sources, or [] if unreachable."""
    try:
        result = api(
            "AcceptanceTestController",
            "getAvailableLightSources",
        )

        return [
            source["name"]
            for source in result.get("light_sources", [])
        ]

    except requests.RequestException:
        return []
LIGHTS = get_lights()

# Only LEDs are asserted on: the laser on this rig is mounted so its beam never
# reaches the sensor, so a photon test on it measures nothing. LIGHTS itself
# stays complete, because all_lights_off() has to switch the lasers off too.
LEDS = [
    name
    for name in LIGHTS
    if "led" in name.lower() and "laser" not in name.lower()
]

# A named skip instead of pytest's generic "got empty parameter set". Applied
# at collection, so the light_source fixture never runs with None.
LED_PARAMS = [pytest.param(name, id=name) for name in LEDS] or [
    pytest.param(
        None,
        id="no-led",
        marks=pytest.mark.skip(
            reason="no LED among the configured light sources"
        ),
    )
]


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


def set_light(name, on):
    """Switch one light source on at LASER_VALUE, or off."""
    api(
        "LaserController",
        "setLaserValue",
        laserName=name,
        value=LASER_VALUE if on else 0,
    )
    api(
        "LaserController",
        "setLaserActive",
        laserName=name,
        active=on,
    )
    time.sleep(SETTLE)


def led_matrix_off():
    """Switch off the LED matrix, which LIGHTS does not cover.

    The matrix sits behind its own controller, which exposes setters only, so
    this writes unconditionally. 404 means the setup has no matrix; any other
    error is one, because a lit matrix invalidates the dark frame.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/api/LEDMatrixController/setAllLEDOff",
            timeout=30,
        )
    except requests.RequestException:
        return

    if response.status_code == 404:
        return

    assert response.status_code == 200, response.text


def all_lights_off():
    """Switch every configured light source off, matrix included."""
    for name in LIGHTS:
        set_light(name, False)

    led_matrix_off()


@pytest.fixture(scope="module", autouse=True)
def camera_acquisition(detector_name, camera_status):
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
        # A long exposure is a refusal rather than a failure: forcing it would
        # make frames slower than SETTLE and the measurement unreliable.
        if status == "long_exposure":
            pytest.skip(
                f"{detector_name}: exposure too long for live view: {started}"
            )

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

    all_lights_off()

    # Only hand back what was taken: a stream that was already running belongs
    # to whoever started it.
    if status == "success":
        api("LiveViewController", "stopLiveView", detectorName=detector_name)


@pytest.fixture(scope="module", params=["acquisition", "observation"])
def detector_name(request):
    """Run every test twice: acquisition detector, then observation camera.

    The observation camera only exists on some rigs, so that parameter skips
    where the setup has none.
    """
    detectors = api("SettingsController", "getDetectorNames")

    if not detectors:
        pytest.skip("no detector configured")

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
            pytest.skip(f"detector {DETECTOR!r} not found")

        return DETECTOR

    # The observation camera has its own parameter, so it is not the fallback.
    acquisition = [
        name for name in detectors if not is_observation_camera(name)
    ]

    if not acquisition:
        pytest.skip("setup has no acquisition detector")

    return acquisition[0]


@pytest.fixture
def light_source(request):
    """Hand over one light source, with everything dark before and after."""
    all_lights_off()
    yield request.param
    all_lights_off()


@pytest.mark.hardware
@pytest.mark.parametrize(
    "light_source",
    LED_PARAMS,
    indirect=True,
)
def test_light_source_is_visible_to_camera(
    light_source,
    detector_name,
    auto_exposure,
    measure_dark_baseline,
    take_image,
    image_difference,
    ):
    """Each LED must change the image by more than the camera noise floor.

    Runs once per camera: the acquisition detector and, where the rig has one,
    the observation camera.
    """
    # Before the baseline, never between the two frames: dark and bright have
    # to share one exposure, or the change is partly the exposure changing.
    #
    # Acquisition detector only: the pass runs once per session, and the
    # observation camera has no live stream for it to act on.
    if not is_observation_camera(detector_name):
        auto_exposure(detector_name)

    dark, noise_floor, required_change = measure_dark_baseline(
        detector_name
    )

    set_light(light_source, True)

    bright = take_image(detector_name)

    change = image_difference(dark, bright)
    print(
        f"\n{light_source}: "
        f"dark_mean={ImageStat.Stat(dark).mean[0]:.2f} "
        f"bright_mean={ImageStat.Stat(bright).mean[0]:.2f} "
        f"noise_floor={noise_floor:.2f} "
        f"pixel_change={change:.2f} "
        f"required={required_change:.2f}"
    )

    assert bright.getextrema()[1] > 0, (
        f"{light_source} is on but every pixel is still 0; "
        f"no light reached the sensor"
    )

    assert change >= required_change, (
        f"{light_source}: image changed only by {change:.2f}; "
        f"camera noise floor is {noise_floor:.2f}, "
        f"need >= {required_change:.2f}"
    )
