"""The report of a run: hil-report.pdf (results, photos, PDFs of the tests) and summary.md.

A test adds to its part of the report with:

    attach("dark", image)                    a photo
    note("The calibration ended as ...")     a line of text
    append_pdf("FRAME calibration", data)    a whole PDF, after the pages of the tests

``__main__.py`` passes ``ReportPlugin`` to pytest,
which collects every result and writes the two files at the end.
Under plain pytest nothing is written.
"""
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

PHOTO_WIDTH = 640  # photos are scaled down to this, so a report with dozens of them stays small
STATUSES = ["failed", "error", "skipped", "passed"]  # the order in the summary

_started = time.time()
_results = {}       # test id -> {"status", "duration", "details"}
_descriptions = {}  # test file -> the first paragraph of its docstring: what it checks
_photos = {}        # test id -> [(label, PNG bytes)]
_notes = {}         # test file -> [lines of text]
_appended = []      # [(title, PDF bytes)]
report_error = None  # why hil-report.pdf could not be written


def reset():
    global _started, report_error
    _started = time.time()
    report_error = None
    _results.clear()
    _descriptions.clear()
    _photos.clear()
    _notes.clear()
    _appended.clear()


def current_test():
    """The id of the test that runs now, "test_x.py::test_y[cam]"; None outside pytest."""
    test = os.environ.get("PYTEST_CURRENT_TEST")  # "test_x.py::test_y[cam] (call)"
    return test.rsplit(" (", 1)[0] if test else None


def attach(label, image):
    """Add a photo to the test that is running. ``image`` is a PIL image or a numpy array."""
    test = current_test()
    if not test:
        return
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    if image.width > PHOTO_WIDTH:
        image = image.resize((PHOTO_WIDTH, round(image.height * PHOTO_WIDTH / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    _photos.setdefault(test, []).append((str(label), buffer.getvalue()))


def note(text):
    """Add a line of text to the part of the test file that is running."""
    test = current_test()
    if test:
        _notes.setdefault(test.split("::")[0], []).append(str(text))


def append_pdf(title, data):
    """Append a PDF (its bytes) to the report, after the pages of the tests."""
    _appended.append((str(title), data))


def record_problem(name, message):
    """Add a problem outside the tests to the results, e.g. a camera that did not start."""
    _results[name] = {"status": "error", "duration": 0.0, "details": str(message)}


def counts():
    """``{"failed": 0, "error": 0, "skipped": 1, "passed": 29}``"""
    statuses = [result["status"] for result in _results.values()]
    return {status: statuses.count(status) for status in STATUSES}


def summary(setup_file):
    """Markdown, short enough for a CI step summary: the setup, the counts, one line per failure."""
    totals = " · ".join(f"{n} {status}" for status, n in counts().items())
    lines = ["### HIL report", "", f"setup `{setup_file}`", "", totals, "",
             "Report: `hil-report.pdf` in the artifact `hil-report`.", ""]
    for test, result in _results.items():
        if result["status"] in ("failed", "error"):
            last_line = (result["details"].strip().splitlines() or [""])[-1]
            lines.append(f"- **{result['status']}** `{test}`: {last_line[:300]}")
    for file, notes in _notes.items():
        for text in notes:
            lines.append(f"- `{file}`: {text}")
    return "\n".join(lines) + "\n"


def write_reports(report_dir, setup_file):
    """Write hil-report.pdf and summary.md. A PDF that fails is a problem in the summary."""
    global report_error
    try:
        # Imported here, so that a missing ReportLab also ends as a problem in the summary.
        from hil_pdf import write_pdf
        write_pdf(report_dir / "hil-report.pdf", setup_file, _started, _results, _descriptions,
                  _photos, _notes, _appended)
    except Exception as error:
        report_error = str(error)
        record_problem("PDF generation", report_error)
        print(f"[hil] PDF generation failed: {error}", flush=True)
    (report_dir / "summary.md").write_text(summary(setup_file), encoding="utf-8")


class ReportPlugin:
    """Collects the results while pytest runs and writes the report files into ``report_dir``.

    ``setup_file`` is the ImSwitch setup the rig ran with; both files name it at the top.
    """

    def __init__(self, report_dir, setup_file):
        self.report_dir = Path(report_dir)
        self.setup_file = setup_file

    def pytest_itemcollected(self, item):
        """Called for every test pytest finds: keep what its file says it checks.

        That is the first paragraph of the file's docstring, its lines joined into one.
        """
        file = item.nodeid.split("::")[0]
        first_paragraph = (item.module.__doc__ or "").strip().split("\n\n")[0]
        _descriptions[file] = " ".join(first_paragraph.split())

    def pytest_runtest_logreport(self, report):
        """Called three times per test: for its setup, the test itself (call) and its teardown."""
        result = _results.setdefault(report.nodeid, {
            "status": "passed", "duration": 0.0, "details": ""})
        result["duration"] += report.duration
        if report.failed and result["status"] != "failed":
            # A failure outside the test itself is a broken fixture, not a failed assertion.
            result["status"] = "failed" if report.when == "call" else "error"
            result["details"] = str(report.longrepr)
        elif report.skipped and result["status"] == "passed":
            result["status"] = "skipped"
            result["details"] = report.longrepr[2]  # (file, line, reason)

    def pytest_sessionfinish(self):
        write_reports(self.report_dir, self.setup_file)
