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
# Not started when a test before it failed: it moves the stage for up to half an hour, and
# without boards, cameras and light it can only fail.
GUARDED = "frame_calibration"

_failed_before = []  # the tests outside GUARDED that failed so far in this run


def name_of(item_or_report):
    """ "board_link" for a test of test_board_link.py."""
    file = item_or_report.nodeid.split("::")[0]  # "test_board_link.py::test_uc2_board_connected"
    return Path(file).stem[len("test_"):]


def pytest_collection_modifyitems(items):
    """Put the tests in ORDER, and mark every HIL test ``hardware``.

    The mark lets ``pytest -m "not hardware"`` leave the suite out.
    """
    # The sort keeps the order of the tests inside a file.
    items.sort(key=lambda item: ORDER.index(name_of(item)) if name_of(item) in ORDER
               else len(ORDER))
    for item in items:
        if Path(__file__).parent in Path(item.fspath).parents:
            item.add_marker(pytest.mark.hardware)


def pytest_runtest_logreport(report):
    """Called with every test's result: remember the failures before GUARDED."""
    if report.failed and name_of(report) != GUARDED:
        _failed_before.append(report.nodeid)


def pytest_runtest_setup(item):
    """Called before a test and its fixtures start: skip GUARDED after a failure."""
    if name_of(item) == GUARDED and _failed_before:
        pytest.skip(f"not started, because {_failed_before[0]} failed")


@pytest.fixture(scope="session", autouse=True)
def imswitch_reachable():
    """Without a server the whole suite skips."""
    if not client.reachable():
        pytest.skip(f"ImSwitch not reachable at {client.IMSWITCH_URL}")
