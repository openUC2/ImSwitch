"""Shared fixtures for the e2e suite: frame capture and photon thresholds."""
import base64
import io
import os
import time

import pytest
import requests
from PIL import Image, ImageChops, ImageStat

BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")

# Must outlast setDetectorExposureOnce's reset timer, or the first frame is
# still taken while the detector is in 'once' mode.
AUTO_EXPOSURE_RESET_MS = int(
    os.environ.get("AUTO_EXPOSURE_RESET_MS", "1500")
)

PHOTON_MIN_DELTA = float(
    os.environ.get("PHOTON_MIN_DELTA", "1.5")
)

PHOTON_NOISE_FACTOR = float(
    os.environ.get("PHOTON_NOISE_FACTOR", "1.5")
)

PHOTON_NOISE_SAMPLES = int(
    os.environ.get("PHOTON_NOISE_SAMPLES", "6")
)

PHOTON_NOISE_DELAY = float(
    os.environ.get("PHOTON_NOISE_DELAY", "0.5")
)

PHOTON_RESIZE = 0.25


def _exposure_ms(detector):
    """Current detector exposure in ms, or None if it cannot be read."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/SettingsController/getCameraStatus",
            params={"detectorName": detector},
            timeout=30,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    parameters = response.json().get("parameters") or {}
    return (parameters.get("exposure") or {}).get("value")

def is_observation_camera(detector):
    """True for the observation camera, which has no acquisition path."""
    return "observ" in detector.lower()


def _snap_observation():
    """One greyscale frame from the observation camera.

    snapNumpyToFastAPI only snaps the current acquisition detector and
    answers 500 for any other, so the frame comes from the overview endpoint.
    Scaled down like the other path, so both cameras are measured at a
    comparable pixel count and one threshold fits both.
    """
    response = requests.post(
        f"{BASE_URL}/api/ExperimentController/snapOverviewImage",
        params={"slot_id": "1", "camera_name": "e2e_photon_test"},
        timeout=60,
    )

    # 400 means ExperimentController has no overview camera bound, even though
    # the setup lists a detector whose name looks like one. That is absent
    # hardware rather than a failure, so it skips like every other such case.
    if response.status_code == 400:
        pytest.skip(f"overview camera not available: {response.text}")

    assert response.status_code == 200, response.text

    frame = Image.open(
        io.BytesIO(base64.b64decode(response.json()["imageBase64"]))
    ).convert("L")

    return frame.resize(
        (
            max(1, int(frame.width * PHOTON_RESIZE)),
            max(1, int(frame.height * PHOTON_RESIZE)),
        )
    )


@pytest.fixture(scope="session")
def take_image():
    """Capture one greyscale camera frame, from either camera path."""

    def run(detector):
        if is_observation_camera(detector):
            return _snap_observation()

        response = requests.get(
            f"{BASE_URL}/api/RecordingController/snapNumpyToFastAPI",
            params={
                "detectorName": detector,
                "resizeFactor": PHOTON_RESIZE,
            },
            timeout=60,
        )

        assert response.status_code == 200, response.text

        return Image.open(
            io.BytesIO(response.content)
        ).convert("L")

    return run


@pytest.fixture(scope="session")
def image_difference():
    """Mean absolute pixel difference between two frames."""

    def run(first, second):
        return ImageStat.Stat(
            ImageChops.difference(first, second)
        ).mean[0]

    return run


@pytest.fixture
def measure_dark_baseline(take_image, image_difference):
    """Measure camera noise, returning (dark frame, noise floor, threshold)."""

    def run(detector):
        previous = take_image(detector)
        drifts = []

        for _ in range(PHOTON_NOISE_SAMPLES - 1):
            time.sleep(PHOTON_NOISE_DELAY)

            current = take_image(detector)

            drift = image_difference(previous, current)
            drifts.append(drift)

            previous = current

        noise_floor = max(drifts)

        required_change = max(
            PHOTON_MIN_DELTA,
            noise_floor * PHOTON_NOISE_FACTOR,
        )

        print(
            f"\n{detector} dark noise: "
            f"drifts={[round(value, 2) for value in drifts]}, "
            f"noise_floor={noise_floor:.2f}, "
            f"required_change={required_change:.2f}"
        )

        return previous, noise_floor, required_change

    return run

@pytest.fixture(scope="session")
def auto_exposure():
    """Run one-shot auto exposure, at most once per pytest session.

    Returns a callable because only the caller knows when live view is
    streaming and the scene is in the state to expose for. The endpoint answers
    200 even on failure, so the exposure before and after is the only evidence.
    """
    state = {}

    def run(detector):
        if "result" in state:
            return state["result"]

        before = _exposure_ms(detector)

        try:
            response = requests.get(
                f"{BASE_URL}/api/SettingsController/setDetectorExposureOnce",
                params={
                    "detectorName": detector,
                    "resetDelayMs": AUTO_EXPOSURE_RESET_MS,
                },
                timeout=30,
            )
            assert response.status_code == 200, response.text

            # Outlast the reset timer, plus a margin for the new exposure to
            # reach the running stream.
            time.sleep(AUTO_EXPOSURE_RESET_MS / 1000 + 0.1)

        except requests.RequestException:
            pass

        after = _exposure_ms(detector)
        state["result"] = (before, after)

        print(
            f"\nauto exposure on {detector}: "
            f"{before} ms -> {after} ms (once per session)"
        )

        return state["result"]

    return run
