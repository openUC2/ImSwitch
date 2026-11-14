"""Run the hardware-in-the-loop (HIL) tests against a live ImSwitch and write a report.

    python -m imswitch.imcontrol._test.hil REPORT_DIR [TEST ...] [--sync-firmware]

    python -m imswitch.imcontrol._test.hil out                           # everything
    python -m imswitch.imcontrol._test.hil out board_link camera_capture # only these tests
    python -m imswitch.imcontrol._test.hil out --sync-firmware           # FLASHES every board first

TEST is the name of a test file without "test_" and ".py": board_link is test_board_link.py.
Without TEST every test runs, which MOVES THE STAGE AND SWITCHES LIGHT.
REPORT_DIR receives junit.xml, report.html and summary.md; both reports name the ImSwitch
setup file (ImSwitchConfig/imcontrol_setups/<name>.json) the rig ran with.

Exit code: 0 all passed, 1 a test failed, 2 the tests could not run.
The test rig (os-rpi) runs exactly this command inside the ImSwitch container.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # so the modules of this folder import as they do under pytest

import pytest  # noqa: E402
import requests  # noqa: E402

import firmware_sync  # noqa: E402
import hil_client as client  # noqa: E402
import hil_report  # noqa: E402

PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2

READY_API_TIMEOUT = 180       # seconds ImSwitch may take to answer
READY_DETECTOR_TIMEOUT = 300  # seconds each camera may take to give its first frame


def can_snap(detector, for_acquisition):
    """Whether the camera gives a frame yet; right after a start its driver may still load."""
    try:
        if for_acquisition:
            return client.get("/RecordingController/snapNumpyToFastAPI", params={
                "detectorName": detector, "resizeFactor": 0.1}).status_code == 200
        # 400: no observation camera is connected. Waiting does not change that, and the
        # tests that need it skip.
        return client.post("/ExperimentController/snapOverviewImage", timeout=30, params={
            "slot_id": "1", "camera_name": "hil_ready_check"}).status_code in (200, 400)
    except requests.RequestException:
        return False


def why_not_ready():
    """Wait for ImSwitch and a readable setup. Returns None, or what is missing."""
    if not client.wait_until(client.reachable, READY_API_TIMEOUT, interval=5):
        return f"ImSwitch did not answer at {client.IMSWITCH_URL} within {READY_API_TIMEOUT}s"
    client.setup()  # A readable setup is needed to collect the tests.
    return None


def prepare_cameras():
    """Give cameras time to start; record failures without blocking calibration."""
    problems = []
    for name, detector in client.setup()["detectors"].items():
        try:
            acquisition = detector["forAcquisition"]
            if acquisition:
                result = client.post_json("/LiveViewController/startLiveView",
                                          {"detectorName": name}, timeout=60)
                assert result.get("status") in ("success", "already_running"), result
            assert client.wait_until(lambda: can_snap(name, acquisition),
                                     READY_DETECTOR_TIMEOUT, interval=5), "no frame before timeout"
        except (requests.RequestException, AssertionError) as error:
            problems.append(f"{name}: {error}")
    return problems



def main():
    parser = argparse.ArgumentParser(prog="python -m imswitch.imcontrol._test.hil")
    parser.add_argument("report_dir", help="folder for junit.xml, report.html and summary.md")
    parser.add_argument("tests", nargs="*",
                        help="tests, e.g. board_link camera_capture; default all")
    parser.add_argument("--sync-firmware", action="store_true",
                        help="FLASH every board to the firmware server's version first")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HIL_REPORT_DIR"] = str(report_dir.resolve())
    hil_report.reset()
    # A reused local report directory must not expose artifacts from an older run.
    for name in ("hil-report.pdf", "test-report.pdf", "calibration-report.pdf",
                 "calibration-status.json", "setup_before_frame_calibration.json", "report-error.txt",
                 "report.html", "summary.md", "junit.xml"):
        (report_dir / name).unlink(missing_ok=True)
    files = [HERE / f"test_{name}.py" for name in args.tests]
    unknown = [name for name, file in zip(args.tests, files) if not file.is_file()]
    if unknown:
        known = " ".join(sorted(file.stem[len("test_"):] for file in HERE.glob("test_*.py")))
        problem = f"no such test: {' '.join(unknown)}. There are: {known}"
        hil_report.record_problem("startup", problem)
        hil_report.write_reports(report_dir, "unavailable")
        return COULD_NOT_RUN

    try:
        problem = why_not_ready()
        if problem is None:
            print(f"[hil] setup: {client.setup_file()}", flush=True)
        if problem is None and args.sync_firmware:
            firmware_sync.sync()
    except (requests.RequestException, AssertionError, firmware_sync.SyncFailed) as error:
        problem = str(error)
    if problem:
        print(f"[hil] not running the tests: {problem}")
        hil_report.record_problem("startup", problem)
        hil_report.write_reports(report_dir, "unavailable")
        return COULD_NOT_RUN

    for problem in prepare_cameras():
        hil_report.record_problem(f"camera preparation: {problem.split(':', 1)[0]}", problem)

    code = pytest.main(
        [*[str(file) for file in files or [HERE]],
         "-c", str(HERE / "pytest.ini"), "-ra", f"--junitxml={report_dir / 'junit.xml'}"],
        plugins=[hil_report.ReportPlugin(report_dir, client.setup_file())])

    counts = hil_report.counts()
    print("[hil] " + " · ".join(f"{n} {status}" for status, n in counts.items()))
    if hil_report.report_error:
        return COULD_NOT_RUN
    if code == pytest.ExitCode.TESTS_FAILED or counts["failed"] or counts["error"]:
        return FAILED
    if code != pytest.ExitCode.OK:
        return COULD_NOT_RUN
    if not counts["passed"]:
        print("[hil] every test skipped: nothing on this rig was actually tested")
        return COULD_NOT_RUN
    return PASSED


if __name__ == "__main__":
    sys.exit(main())
