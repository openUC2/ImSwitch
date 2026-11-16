"""Offline checks of HIL orchestration and printable reports; never contact a rig."""
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfReader
from reportlab.pdfgen.canvas import Canvas

HIL = Path(__file__).parents[1] / "hil"
sys.path.insert(0, str(HIL))
import hil_report
import hil_pdf


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def calibration_module():
    hil_report.reset()
    return load("hil_calibration_test", HIL / "test_frame_calibration.py")


def pdf_bytes(text="Calibration evidence"):
    data = io.BytesIO()
    canvas = Canvas(data)
    canvas.drawString(40, 700, text)
    canvas.save()
    return data.getvalue()


def test_pdf_has_the_failure_the_note_and_the_appended_pdf(tmp_path):
    results = {"test_camera.py::test_capture": {
        "status": "failed", "duration": 2., "details": "Diagnostic line\n" * 140 + "E   LAST ERROR LINE"}}
    hil_pdf.write_pdf(tmp_path / "hil-report.pdf", "FRAME.json", 0, results,
                      {"test_camera.py": "Every camera returns a frame."}, {},
                      {"test_camera.py": ["A NOTE OF THE TEST"]},
                      [("Calibration", pdf_bytes())])
    reader = PdfReader(tmp_path / "hil-report.pdf")
    text = "\n".join(page.extract_text() for page in reader.pages[:-1])
    assert "LAST ERROR LINE" in text and "Every camera returns a frame." in text
    assert "A NOTE OF THE TEST" in text
    assert "Diagnostic line" not in text  # only what went wrong, not the whole traceback
    assert "Calibration evidence" in reader.pages[-1].extract_text()
    assert len(reader.outline) == 2
    assert [file.name for file in tmp_path.iterdir()] == ["hil-report.pdf"]


def test_what_a_test_checks_is_the_whole_first_paragraph_of_its_docstring():
    hil_report.reset()
    module = SimpleNamespace(__doc__="Every axis moves 10 um\nand steps back.\n\nMOVES HARDWARE.")
    hil_report.ReportPlugin("out", "FRAME.json").pytest_itemcollected(
        SimpleNamespace(nodeid="test_motor.py::test_it", module=module))
    assert hil_report._descriptions["test_motor.py"] == "Every axis moves 10 um and steps back."


def test_report_without_appended_pdf_has_only_the_test_pages(tmp_path):
    hil_report.reset()
    hil_report.record_problem("startup", "server unavailable")
    hil_report.write_reports(tmp_path, "unavailable")
    reader = PdfReader(tmp_path / "hil-report.pdf")
    assert "server unavailable" in "".join(page.extract_text() for page in reader.pages)
    assert len(reader.outline) == 1
    assert sorted(file.name for file in tmp_path.iterdir()) == ["hil-report.pdf", "summary.md"]


def test_pdf_failure_still_writes_the_summary(tmp_path, monkeypatch):
    hil_report.reset()
    def fail(*args):
        raise ValueError("broken PDF")
    monkeypatch.setattr(hil_pdf, "write_pdf", fail)
    hil_report.write_reports(tmp_path, "FRAME.json")
    assert hil_report.report_error == "broken PDF"
    assert not (tmp_path / "hil-report.pdf").exists()
    assert "broken PDF" in (tmp_path / "summary.md").read_text()
    assert "hil-report.pdf` in the artifact" not in (tmp_path / "summary.md").read_text()


def test_a_long_image_list_fits_on_the_pages(tmp_path, monkeypatch):
    # Run 43896090621: the pallet text in one table cell was taller than a page (LayoutError).
    images = [f"ghcr.io/openuc2/image-{number}:sha-{number:07d}" for number in range(200)]
    monkeypatch.setenv("HIL_IMAGES", "\n".join(images))
    results = {"test_board_link.py::test_it": {"status": "passed", "duration": 1., "details": ""}}
    hil_pdf.write_pdf(tmp_path / "hil-report.pdf", "FRAME.json", 0, results, {}, {}, {}, [])
    text = "".join(page.extract_text() for page in PdfReader(tmp_path / "hil-report.pdf").pages)
    assert images[0] in text and images[-1] in text


def test_a_long_output_fits_on_the_pages(tmp_path):
    # Many failed cases of one file: their lines must not end up in one cell taller than a page.
    results = {f"test_motor.py::test_it[{axis}{number}]": {
        "status": "failed", "duration": 1., "details": "E   moved by 0\n" * 12}
        for axis in "XYZA" for number in range(10)}
    hil_pdf.write_pdf(tmp_path / "hil-report.pdf", "FRAME.json", 0, results, {}, {}, {}, [])
    assert len(PdfReader(tmp_path / "hil-report.pdf").pages) > 2


def test_summary_has_one_row_per_test_file():
    hil_report.reset()
    hil_report._results.update({
        "test_board_link.py::test_a": {"status": "passed", "duration": 1., "details": ""},
        "test_lightsource.py::test_a[LED]": {"status": "failed", "duration": 2.,
                                             "details": "E   LED: not seen"},
        "test_lightsource.py::test_b[LED]": {"status": "passed", "duration": 3., "details": ""}})
    lines = hil_report.summary("FRAME.json").splitlines()
    assert "| 1 | board_link | passed | 1 passed | 1 s |" in lines
    assert "| 2 | lightsource | **failed** | 1 failed, 1 passed | 5 s |" in lines
    assert any("LED: not seen" in line for line in lines)


def test_stop_timeout_is_an_error(calibration_module, monkeypatch):
    m = calibration_module
    monkeypatch.setattr(m, "running", lambda: True)
    monkeypatch.setattr(m, "status", lambda: {"state": "finished"})
    monkeypatch.setattr(m.client, "wait_until", lambda *a, **k: False)
    with pytest.raises(AssertionError, match="thread did not stop"):
        m.stop_and_wait()


def test_poll_waits_for_thread_after_terminal_status(calibration_module, monkeypatch):
    m = calibration_module
    monkeypatch.setattr(m.client, "post_json", lambda *a, **k: {"success": True})
    monkeypatch.setattr(m, "status", lambda: {"state": "finished", "startedAt": "now", "steps": []})
    states = iter([True, False])
    monkeypatch.setattr(m, "running", lambda: next(states))
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    run = {}
    assert m.run_calibration(run, lambda line: None)["state"] == "finished"
    assert run == {"accepted": True, "startedAt": "now"}


def test_export_rejects_other_run(calibration_module, monkeypatch):
    m = calibration_module
    monkeypatch.setattr(m, "running", lambda: False)
    monkeypatch.setattr(m, "status", lambda: {"startedAt": "older"})
    with pytest.raises(AssertionError, match="identity changed"):
        m.collect_report({"startedAt": "ours"})


def test_failed_calibration_pdf_is_collected(calibration_module, monkeypatch, tmp_path):
    m = calibration_module
    monkeypatch.setenv("HIL_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(m, "running", lambda: False)
    monkeypatch.setattr(m, "status", lambda: {"startedAt": "ours", "state": "failed", "message": "no focus"})
    monkeypatch.setattr(m.client, "get", lambda *a, **k: SimpleNamespace(status_code=200, content=pdf_bytes()))
    m.collect_report({"startedAt": "ours"})
    (title, data), = hil_report._appended
    assert data.startswith(b"%PDF-")
    (notes,) = hil_report._notes.values()
    assert notes[0] == "The calibration ended as failed: no focus"
    assert "failed: no focus" in hil_report.summary("FRAME.json")
    # Also saved on its own, so it survives a hil-report.pdf that cannot be written.
    assert (tmp_path / "frame_calibration.pdf").read_bytes() == data


def test_calibration_progress_is_one_line_per_start_and_end(calibration_module):
    steps = [{"name": "homing", "title": "Home", "status": "ok", "detail": "homed Z", "seconds": 12.4,
              "warnings": ["slow"]},
             {"name": "objective0", "title": "First objective", "status": "running", "detail": "",
              "seconds": None, "warnings": []},
             {"name": "save", "title": "Save", "status": "pending", "detail": "", "seconds": None,
              "warnings": []}]
    lines, shown = [], set()
    calibration_module.show_progress(steps, shown, lines.append)
    calibration_module.show_progress(steps, shown, lines.append)  # nothing changed: no new line
    assert lines == ["calibration  1/3  Home ...", "calibration  1/3  ok (12 s) homed Z",
                     "                  warning: slow", "calibration  2/3  First objective ..."]


def test_tests_run_in_order_and_unlisted_ones_last():
    conf = load("hil_hooks", HIL / "conftest.py")
    items = [SimpleNamespace(nodeid=f"test_{name}.py::test_it", fspath=HIL / f"test_{name}.py",
                             add_marker=lambda _: None)
             for name in ("frame_calibration", "new_test", "motor_axis_move", "board_link")]
    conf.pytest_collection_modifyitems(items)
    assert [conf.name_of(item) for item in items] == [
        "board_link", "motor_axis_move", "frame_calibration", "new_test"]


def test_calibration_summary_overflow_keeps_last_step(tmp_path):
    report = load("frame_pdf", HIL.parents[1] / "controller/controllers/frame_calibration_report.py")
    status = {"state": "failed", "steps": [dict(title=f"Step {i}", status="failed", detail="long detail " * 100,
                                               warnings=["warning " * 50]) for i in range(12)]}
    path = tmp_path / "calibration.pdf"
    report.write_pdf(path, status, {"target": "plate"}, [])
    reader = PdfReader(path)
    assert len(reader.pages) > 1
    assert "Step 11" in "".join(p.extract_text() for p in reader.pages)


def test_camera_preparation_failure_does_not_prevent_test_execution(tmp_path, monkeypatch):
    main = load("hil_main", HIL / "__main__.py")
    monkeypatch.setattr(sys, "argv", ["hil", str(tmp_path)])
    monkeypatch.setattr(main, "wait_for_imswitch", lambda: None)
    monkeypatch.setattr(main, "prepare_cameras", lambda: {"camera A": "no frame"})
    monkeypatch.setattr(main.client, "setup_file", lambda: "FRAME.json")
    called = []
    def tests(*args, **kwargs):
        called.append(True)
        return pytest.ExitCode.OK
    monkeypatch.setattr(main.pytest, "main", tests)
    assert main.main() == main.FAILED
    assert called == [True]
    assert hil_report.counts()["error"] == 1


def test_unreachable_server_still_writes_report(tmp_path, monkeypatch):
    main = load("hil_main", HIL / "__main__.py")
    (tmp_path / "hil-report.pdf").write_bytes(pdf_bytes("OLD RUN"))
    monkeypatch.setattr(sys, "argv", ["hil", str(tmp_path)])
    def unavailable():
        raise AssertionError("server unavailable")
    monkeypatch.setattr(main, "wait_for_imswitch", unavailable)
    assert main.main() == main.COULD_NOT_RUN
    text = "".join(page.extract_text() for page in PdfReader(tmp_path / "hil-report.pdf").pages)
    assert "server unavailable" in text and "OLD RUN" not in text


def test_failed_calibration_is_stopped_and_its_report_collected(calibration_module, monkeypatch):
    m = calibration_module
    called = []
    def fails(run, live):
        run.update(accepted=True, startedAt="ours")
        raise AssertionError("original failure")
    monkeypatch.setattr(m, "run_calibration", fails)
    monkeypatch.setattr(m, "stop_and_wait", lambda: called.append("stop"))
    monkeypatch.setattr(m, "collect_report", lambda run: called.append("pdf"))
    monkeypatch.setattr(m, "all_lights_off", lambda: called.append("off"))

    # As pytest does it: the fixtures start in the order of ``calibration``'s arguments,
    # and end in the reverse order.
    run_fixture = m.run.__wrapped__()
    lights_fixture = m.lights_off_afterwards.__wrapped__()
    run = next(run_fixture)
    next(lights_fixture)
    with pytest.raises(AssertionError, match="original failure"):
        m.calibration.__wrapped__(lambda line: None, run, None, None)
    for fixture in (lights_fixture, run_fixture):
        with pytest.raises(StopIteration):
            next(fixture)
    assert called == ["stop", "off", "pdf"]


def test_a_refused_calibration_is_not_stopped_and_gives_no_report(calibration_module, monkeypatch):
    m = calibration_module
    def refused(run, live):
        raise AssertionError("startFrameCalibration refused")
    monkeypatch.setattr(m, "run_calibration", refused)
    monkeypatch.setattr(m, "stop_and_wait", lambda: pytest.fail("stopped another run"))
    monkeypatch.setattr(m, "collect_report", lambda run: pytest.fail("took another run's report"))
    run_fixture = m.run.__wrapped__()
    run = next(run_fixture)
    with pytest.raises(AssertionError, match="refused"):
        m.calibration.__wrapped__(lambda line: None, run, None, None)
    with pytest.raises(StopIteration):
        next(run_fixture)


@pytest.mark.parametrize("failing, watched, runs", [
    ("firmware_server", "frame_calibration", True),   # another firmware version can still calibrate
    ("board_link", "frame_calibration", False),
    ("motor_axis_move", "frame_calibration", False),
])
def test_a_test_does_not_start_after_a_failure_it_depends_on(tmp_path, failing, watched, runs):
    import os
    import subprocess
    conf = (HIL / "conftest.py").read_text()
    (tmp_path / "conftest.py").write_text(conf + "\nclient.reachable = lambda: True\n")
    (tmp_path / f"test_{failing}.py").write_text("def test_it():\n    assert False, 'offline simulated failure'\n")
    (tmp_path / f"test_{watched}.py").write_text(
        "from pathlib import Path\ndef test_it(live):\n    live('LIVE LINE')\n"
        "    Path(__file__).with_suffix('.ran').touch()\n")
    env = dict(os.environ, PYTHONPATH=str(HIL), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run([sys.executable, "-m", "pytest", str(tmp_path), "--confcutdir", str(tmp_path), "-q", "-rs"],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / f"test_{watched}.ran").exists() == runs
    # live() prints although pytest captures the output of a passing test
    assert ("LIVE LINE" in result.stdout) == runs
    if not runs:
        assert f"not started, because {failing} failed" in result.stdout
