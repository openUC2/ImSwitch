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
    "motor_axis_move",                               # 3. the motors: 10 µm steps,
    "stage_homing",                                  #    homing, ending at the transport position,
    "motor_motion_camera",                           #    then moves the camera sees
    "lightsource", "ledmatrix", "ledmatrix_single",  # 4. the lights, at the transport position
    "frame_calibration",                             # 5. the calibration
]
# A test does not start when one of these failed before it.
# firmware_server is in no list: a board on another firmware version fails that test,
# but can still move and be calibrated.
# Only --sync-firmware stops every test, when flashing fails (__main__.py).
NEEDS = {
    "stage_homing": ["motor_axis_move"],
    # Its first move is absolute, so the stage must be homed.
    "motor_motion_camera": ["motor_axis_move", "stage_homing"],
    "frame_calibration": ["board_link", "camera_capture", "motor_axis_move", "stage_homing",
                          "motor_motion_camera", "lightsource", "ledmatrix", "ledmatrix_single"],
}

_failed = []  # the names of the tests that failed so far in this run


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
    """Called with every test's result: remember the tests that failed."""
    if report.failed:
        _failed.append(name_of(report))


def pytest_runtest_setup(item):
    """Called before a test and its fixtures start: skip it when a test it needs failed."""
    failed = [name for name in NEEDS.get(name_of(item), []) if name in _failed]
    if failed:
        pytest.skip(f"not started, because {failed[0]} failed")


@pytest.fixture(scope="session", autouse=True)
def imswitch_reachable():
    """Without a server the whole suite skips."""
    if not client.reachable():
        pytest.skip(f"ImSwitch not reachable at {client.IMSWITCH_URL}")


@pytest.fixture(scope="session")
def live(request):
    """``live("step 4/10 ...")`` prints a line at once, for a test that runs for minutes.

    pytest keeps what a test prints and shows it only when the test fails.
    This switches that off for the one line.
    """
    capture = request.config.pluginmanager.getplugin("capturemanager")
    reporter = request.config.pluginmanager.getplugin("terminalreporter")

    def show(text):
        with capture.global_and_fixture_disabled():
            reporter.ensure_newline()  # pytest has written the test's name without a line end
            reporter.write_line(text)
    return show
