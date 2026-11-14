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


def test_pdf_merge_keeps_original_pages_and_long_failures(tmp_path):
    original = tmp_path / "calibration-report.pdf"
    original.write_bytes(pdf_bytes())
    results = {"test_camera.py::test_capture": {
        "status": "failed", "duration": 2., "details": "Diagnostic line\n" * 140 + "LAST ERROR LINE", "output": ""}}
    hil_pdf.write_pdf(tmp_path, "FRAME.json", results, {},
                      {"state": "finished", "pdf": str(original)}, 0)
    reader = PdfReader(tmp_path / "hil-report.pdf")
    assert len(reader.pages) > 3
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "LAST ERROR LINE" in text and "TEST RESULT: FAILED" in text
    assert "Calibration evidence" in reader.pages[-1].extract_text()
    assert len(reader.outline) == 2


def test_old_pdf_is_not_appended_without_current_run(tmp_path):
    (tmp_path / "calibration-report.pdf").write_bytes(pdf_bytes("OLD RUN"))
    hil_report.reset()
    hil_report.record_problem("startup", "server unavailable")
    hil_report.write_reports(tmp_path, "unavailable")
    text = "".join(p.extract_text() for p in PdfReader(tmp_path / "hil-report.pdf").pages)
    assert "OLD RUN" not in text and "No calibration PDF" in text


def test_pdf_failure_preserves_other_reports_and_removes_stale_pdf(tmp_path, monkeypatch):
    hil_report.reset()
    (tmp_path / "hil-report.pdf").write_bytes(pdf_bytes("OLD RUN"))
    def fail(*args):
        raise ValueError("broken PDF")
    monkeypatch.setattr(hil_pdf, "write_pdf", fail)
    hil_report.write_reports(tmp_path, "FRAME.json")
    assert hil_report.report_error == "broken PDF"
    assert not (tmp_path / "hil-report.pdf").exists()
    assert "broken PDF" in (tmp_path / "summary.md").read_text()
    assert (tmp_path / "report.html").exists()


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
    assert m.run_calibration(run)["state"] == "finished"
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
    assert hil_report.calibration["state"] == "failed"
    assert (tmp_path / "calibration-report.pdf").exists()


def test_calibration_is_last_without_failure_guard():
    conf = load("hil_hooks", HIL / "conftest.py")
    items = [SimpleNamespace(nodeid=f"test_{name}.py::test_it", fspath=HIL / f"test_{name}.py",
                             add_marker=lambda _: None) for name in ("frame_calibration", "new_test", "board_link")]
    conf.pytest_collection_modifyitems(items)
    assert conf.name_of(items[-1]) == "frame_calibration"
    assert not hasattr(conf, "pytest_runtest_setup")


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
    monkeypatch.setattr(main, "why_not_ready", lambda: None)
    monkeypatch.setattr(main, "prepare_cameras", lambda: ["camera A: no frame"])
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
    (tmp_path / "calibration-report.pdf").write_bytes(pdf_bytes("old calibration"))
    (tmp_path / "report-error.txt").write_text("old error")
    monkeypatch.setattr(sys, "argv", ["hil", str(tmp_path)])
    monkeypatch.setattr(main, "why_not_ready", lambda: "server unavailable")
    assert main.main() == main.COULD_NOT_RUN
    assert (tmp_path / "hil-report.pdf").exists()
    assert not (tmp_path / "calibration-report.pdf").exists()
    assert not (tmp_path / "report-error.txt").exists()


def test_cleanup_error_does_not_prevent_lights_off_or_export(calibration_module, tmp_path, monkeypatch):
    m = calibration_module
    monkeypatch.setenv("HIL_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(m.client, "setup", lambda: {"positioners": {"stage": {}}, "lasers": {}, "LEDMatrixs": {}})
    monkeypatch.setattr(m.client, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    monkeypatch.setattr(m.client, "json_of", lambda _: {"target": "plate", **m.TEST_SETTINGS})
    monkeypatch.setattr(m, "running", lambda: False)
    monkeypatch.setattr(m, "save_setup_backup", lambda: None)
    monkeypatch.setattr(m, "set_settings", lambda _: None)
    called = []
    def run(ctx):
        ctx.update(accepted=True, startedAt="ours")
        raise AssertionError("original failure")
    def stop():
        raise AssertionError("stop timeout")
    monkeypatch.setattr(m, "run_calibration", run)
    monkeypatch.setattr(m, "stop_and_wait", stop)
    monkeypatch.setattr(m, "all_lights_off", lambda: called.append("off"))
    monkeypatch.setattr(m, "collect_report", lambda _: called.append("pdf"))
    monkeypatch.setattr(m, "attach_figures", lambda _: called.append("figures"))
    with pytest.raises(RuntimeError, match="original failure.*stop timeout"):
        m.calibration.__wrapped__()
    assert called == ["off", "pdf", "figures"]


def test_pytest_continues_to_calibration_after_failure(tmp_path):
    import os
    import subprocess
    conf = (HIL / "conftest.py").read_text()
    (tmp_path / "conftest.py").write_text(conf + "\nclient.reachable = lambda: True\n")
    (tmp_path / "test_board_link.py").write_text("def test_board():\n    assert False, 'offline simulated failure'\n")
    (tmp_path / "test_frame_calibration.py").write_text(
        "from pathlib import Path\ndef test_calibration():\n    Path(__file__).with_suffix('.ran').touch()\n")
    env = dict(os.environ, PYTHONPATH=str(HIL), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run([sys.executable, "-m", "pytest", str(tmp_path), "--confcutdir", str(tmp_path), "-q"],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / "test_frame_calibration.ran").exists()
