"""What applies to every HIL test: the order they run in, and when they do not run at all.

Nothing here knows what a test checks.
"""
from pathlib import Path

import pytest

import hil_client as client

# The order the test files run in. A test file that is not listed runs after these.
ORDER = [
    "board_link", "firmware_server",                 # 1. the boards and their firmware versions
    "camera_capture",                                # 2. the cameras
    "lightsource", "ledmatrix", "ledmatrix_single",  # 3. the lights
    "frame_calibration",                             # 4. the calibration: moves every motor
]

def name_of(item_or_report):
    """ "board_link" for a test of test_board_link.py."""
    file = item_or_report.nodeid.split("::")[0]  # "test_board_link.py::test_uc2_board_connected"
    return Path(file).stem[len("test_"):]


def pytest_collection_modifyitems(items):
    """Put the tests in ORDER, and mark every HIL test ``hardware``.

    The mark lets ``pytest -m "not hardware"`` leave the suite out.
    """
    # The sort keeps the order of the tests inside a file.
    items.sort(key=lambda item: (name_of(item) == "frame_calibration",
                                ORDER.index(name_of(item)) if name_of(item) in ORDER else len(ORDER)))
    for item in items:
        if Path(__file__).parent in Path(item.fspath).parents:
            item.add_marker(pytest.mark.hardware)


@pytest.fixture(scope="session", autouse=True)
def imswitch_reachable():
    """Without a server the whole suite skips."""
    if not client.reachable():
        pytest.skip(f"ImSwitch not reachable at {client.IMSWITCH_URL}")
