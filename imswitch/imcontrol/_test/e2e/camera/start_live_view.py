"""Start the live view for every acquisition camera. Not a test.

MOVES NOTHING. Named without the test_ prefix so pytest never collects it; the
runners call it before pytest, the way they call move_to_transport.py.

Why it is needed: snapNumpyToFastAPI does not start anything. It reads the
newest frame out of a ring buffer that only fills while an acquisition loop
runs, waits a second when that buffer is empty and then answers 500. On the
Hikrobot camera that loop is started by the live view, so after a container or
image restart the camera tests fail until something opens a stream.

The observation camera is skipped: it is no acquisition detector and is read
straight off the device through the overview endpoint (see conftest.py).

    IMSWITCH_URL=http://<pi>:8000/imswitch python3 start_live_view.py
"""

import os
import sys

import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")


def is_observation_camera(detector):
    """True for the observation camera, which has no acquisition path."""
    return "observ" in detector.lower()


def start(detector):
    """Start one live view. True when a stream is running afterwards.

    startLiveView answers 200 even when it declines, so the body decides.
    "already_running" is as good as "success": what matters is that a stream
    exists, not who started it.
    """
    response = requests.post(
        f"{BASE_URL}/api/LiveViewController/startLiveView",
        params={"detectorName": detector},
        timeout=60,
    )

    if response.status_code != 200:
        print(f"{detector}: startLiveView -> HTTP {response.status_code}")
        return False

    status = response.json().get("status")
    print(f"{detector}: live view {status}")

    return status in ("success", "already_running")


def main():
    try:
        detectors = requests.get(
            f"{BASE_URL}/api/SettingsController/getDetectorNames", timeout=15
        ).json()
    except requests.RequestException as exc:
        print(f"ImSwitch not reachable at {BASE_URL}: {exc}")
        return 1

    acquisition = [name for name in detectors if not is_observation_camera(name)]

    if not acquisition:
        print(f"no acquisition detector among {detectors}, nothing to start")
        return 0

    # Report a refusal, but do not stop the run: the tests themselves are the
    # better place to see what a camera is really doing.
    return 0 if all([start(name) for name in acquisition]) else 1


if __name__ == "__main__":
    sys.exit(main())
