"""The report of a run: report.html (results, output and photos in one file) and summary.md.

A test adds a photo with ``attach("dark", image)``. ``__main__.py`` passes ``ReportPlugin`` to
pytest, which collects every result and writes the two files at the end. Under plain pytest
nothing is written.
"""
import base64
import html
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

PHOTO_WIDTH = 640  # photos are scaled down to this, so a report with dozens of them stays small
STATUSES = ["failed", "error", "skipped", "passed"]  # the order in the report

_started = time.time()
_photos = {}   # test id -> [(label, base64 PNG)]
_results = {}  # test id -> {"status", "duration", "details", "output"}


def attach(label, image):
    """Add a photo to the test that is running. ``image`` is a PIL image or a numpy array."""
    test = os.environ.get("PYTEST_CURRENT_TEST")  # "camera/test_x.py::test_y[cam] (call)"
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
    photo = base64.b64encode(buffer.getvalue()).decode("ascii")
    _photos.setdefault(test.rsplit(" (", 1)[0], []).append((str(label), photo))


def counts():
    """``{"failed": 0, "error": 0, "skipped": 1, "passed": 29}``"""
    statuses = [result["status"] for result in _results.values()]
    return {status: statuses.count(status) for status in STATUSES}


class ReportPlugin:
    """Collects the results while pytest runs and writes the report files into ``report_dir``.

    ``setup_file`` is the ImSwitch setup the rig ran with; both files name it at the top.
    """

    def __init__(self, report_dir, setup_file):
        self.report_dir = Path(report_dir)
        self.setup_file = setup_file

    def pytest_runtest_logreport(self, report):
        """Called three times per test: for its setup, the test itself (call) and its teardown."""
        result = _results.setdefault(report.nodeid, {
            "status": "passed", "duration": 0.0, "details": "", "output": ""})
        result["duration"] += report.duration
        result["output"] = report.capstdout or result["output"]
        if report.failed and result["status"] != "failed":
            # A failure outside the test itself is a broken fixture, not a failed assertion.
            result["status"] = "failed" if report.when == "call" else "error"
            result["details"] = str(report.longrepr)
        elif report.skipped and result["status"] == "passed":
            result["status"] = "skipped"
            result["details"] = report.longrepr[2]  # (file, line, reason)

    def pytest_sessionfinish(self):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / "summary.md").write_text(summary(self.setup_file), encoding="utf-8")
        (self.report_dir / "report.html").write_text(page(self.setup_file), encoding="utf-8")


def summary(setup_file):
    """Markdown, short enough for a CI step summary: the setup, the counts, one line per failure."""
    totals = " · ".join(f"{n} {status}" for status, n in counts().items())
    lines = ["### HIL report", "", f"setup `{setup_file}`", "", totals, ""]
    for test, result in _results.items():
        if result["status"] in ("failed", "error"):
            last_line = (result["details"].strip().splitlines() or [""])[-1]
            lines.append(f"- **{result['status']}** `{test}`: {last_line[:300]}")
    return "\n".join(lines) + "\n"


def page(setup_file):
    """The HTML report: failures first, then skips, then passes."""
    tests = sorted(_results.items(), key=lambda item: STATUSES.index(item[1]["status"]))
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_started))
    totals = "".join(f'<span style="color:var(--{status})">{n} {status}</span>'
                     for status, n in counts().items())
    return PAGE.format(css=CSS, setup=html.escape(setup_file), started=started,
                       took=time.time() - _started, totals=totals,
                       tests="\n".join(card(test, result) for test, result in tests))


def card(test, result):
    photos = "".join(PHOTO.format(label=html.escape(label), data=data)
                     for label, data in _photos.get(test, []))
    details = f"<pre>{html.escape(result['details'])}</pre>" if result["details"] else ""
    output = OUTPUT.format(output=html.escape(result["output"])) if result["output"].strip() else ""
    return CARD.format(status=result["status"], test=html.escape(test), details=details,
                       duration=result["duration"], photos=photos, output=output)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HIL report</title><style>{css}</style></head><body>
<h1>HIL report</h1>
<div class="meta">setup <code>{setup}</code> · started {started} · took {took:.0f} s</div>
<div class="counts">{totals}</div>
{tests}
</body></html>
"""

CARD = """<section class="test" style="border-left-color:var(--{status})">
<h2><span class="badge" style="color:var(--{status})">{status}</span>{test}
<span class="duration">{duration:.1f} s</span></h2>
{details}
<div class="photos">{photos}</div>
{output}
</section>"""

PHOTO = ('<figure><img alt="{label}" src="data:image/png;base64,{data}">'
         '<figcaption>{label}</figcaption></figure>')

OUTPUT = "<details><summary>output</summary><pre>{output}</pre></details>"

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --card:#f5f5f7; --line:#d2d2d7;
        --passed:#1a7f37; --failed:#cf222e; --error:#bc4c00; --skipped:#6e6e73; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161618; --fg:#f5f5f7; --muted:#a1a1a6; --card:#222225; --line:#3a3a3d;
          --passed:#3fb950; --failed:#f85149; --error:#db6d28; --skipped:#a1a1a6; } }
body { background:var(--bg); color:var(--fg); margin:0 auto; max-width:1100px;
       padding:24px 16px; font:15px/1.5 system-ui, sans-serif; }
h1 { font-size:22px; margin:0 0 4px; }
.meta { color:var(--muted); margin-bottom:16px; }
.counts span { margin-right:16px; font-weight:600; }
.test { background:var(--card); border:1px solid var(--line); border-left-width:6px;
        border-radius:8px; padding:12px 16px; margin:12px 0; }
.test h2 { font-size:15px; margin:0; word-break:break-all; }
.badge { display:inline-block; min-width:64px; font-size:12px; font-weight:700;
         text-transform:uppercase; margin-right:8px; }
.duration { color:var(--muted); font-weight:400; margin-left:8px; }
pre { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px;
      overflow-x:auto; font-size:12.5px; white-space:pre-wrap; }
summary { cursor:pointer; color:var(--muted); margin-top:8px; }
.photos { display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; }
figure { margin:0; max-width:100%; }
figure img { max-width:min(100%, 480px); border:1px solid var(--line); border-radius:4px;
             display:block; }
figcaption { color:var(--muted); font-size:13px; }
"""
