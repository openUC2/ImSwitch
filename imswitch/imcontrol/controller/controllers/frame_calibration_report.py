"""Report figures for the FRAME calibration: annotated images (expected in yellow,
measured in green), the focus curve, and the downloadable PDF / ZIP.

Hardware-free (numpy, OpenCV, matplotlib); used by FrameCalibrationController.
Self-check: python -m imswitch.imcontrol.controller.controllers.frame_calibration_report
"""
import io
import os
import textwrap
import zipfile

import cv2
import numpy as np

EXPECTED = (0, 215, 255)     # BGR yellow
MEASURED = (0, 220, 0)       # BGR green
OTHER = (255, 200, 0)        # BGR cyan-ish
WIDTH = 960


def to_display(img):
    """Raw frame (12/16-bit mono, 8-bit RGB) -> 8-bit BGR for viewing: linear up to the
    99.9th percentile, square-root stretched when the frame is mostly dark."""
    img = np.asarray(img)
    if img.ndim == 3 and img.dtype == np.uint8:
        return img[..., :3].copy()
    g = img.astype(np.float32) if img.ndim == 2 else img.astype(np.float32).mean(axis=2)
    lo, hi = float(np.percentile(g, 0.1)), float(np.percentile(g, 99.9))
    x = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    if np.median(x) < 0.05:
        x = np.sqrt(x)
    return cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def annotate(img, marks=(), width=WIDTH):
    """Draw marks on a display copy of ``img`` scaled to ``width`` px (marks in source px).

    marks: ("circle", (x, y), r, color[, dashed]) | ("cross", (x, y), size, color)
           | ("line", (x1, y1), (x2, y2), color) | ("arrow", (x1, y1), (x2, y2), color)
           | ("text", (x, y), "label", color)
    Labels are ASCII (OpenCV fonts); the explanations go next to the image."""
    out = to_display(img)
    f = width / out.shape[1]
    out = cv2.resize(out, (width, int(round(out.shape[0] * f))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    P = lambda p: (int(round(p[0] * f)), int(round(p[1] * f)))
    th = 2
    for m in marks:
        kind = m[0]
        if kind == "circle":
            c, r, col = P(m[1]), max(2, int(round(m[2] * f))), m[3]
            if len(m) > 4 and m[4]:          # dashed
                for a in range(0, 360, 20):
                    cv2.ellipse(out, c, (r, r), 0, a, a + 10, col, th)
            else:
                cv2.circle(out, c, r, col, th)
        elif kind == "cross":
            c, s, col = P(m[1]), max(4, int(round(m[2] * f))), m[3]
            cv2.line(out, (c[0] - s, c[1]), (c[0] + s, c[1]), col, th)
            cv2.line(out, (c[0], c[1] - s), (c[0], c[1] + s), col, th)
        elif kind == "line":
            cv2.line(out, P(m[1]), P(m[2]), m[3], th)
        elif kind == "arrow":
            cv2.arrowedLine(out, P(m[1]), P(m[2]), m[3], th, tipLength=0.15)
        elif kind == "text":
            p = P(m[1])
            cv2.putText(out, str(m[2]), p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(out, str(m[2]), p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, m[3], 1)
    return out


def montage(images, cols=2, width=WIDTH):
    """Tile already annotated BGR images into one figure of ``width`` px."""
    w = width // cols
    tiles = [cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1])))) for im in images]
    h = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles[i:i + cols]]
        row += [np.zeros((h, w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def plot_curves(curves, xlabel, ylabel, mark_x=None, width=WIDTH):
    """Line plot (BGR image) of [(xs, ys, label), ...] with an optional vertical marker."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width / 100, width / 250), dpi=100)
    ax = fig.add_subplot(111)
    for xs, ys, label in curves:
        ax.plot(xs, ys, "o-", ms=3, label=label)
    if mark_x is not None:
        ax.axvline(mark_x, color="tab:green", lw=1.5, label=f"chosen {mark_x:.0f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def jpeg(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def write_pdf(path, status, settings, figures):
    """Report PDF: run summary and steps, then every figure with what was expected and
    what was measured. ``figures``: dicts with title/step/expected/measured/ok and the
    JPEG bytes under "jpg"."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    def wrap(s, n=110):
        return "\n".join(textwrap.fill(line, n) for line in str(s).splitlines() or [""])

    with PdfPages(path) as pdf:
        fig = Figure(figsize=(8.27, 11.69))
        y = 0.96
        fig.text(0.06, y, "FRAME calibration report", fontsize=16, weight="bold")
        y -= 0.03
        res = status.get("result") or {}
        cam = status.get("camera") or {}
        for line in (f"state: {status.get('state')}   started {status.get('startedAt')}   finished {status.get('finishedAt')}",
                     wrap(status.get("message", ""), 100), f"report folder: {status.get('reportDir', '')}",
                     f"stored calibration: {res.get('timestamp', 'none')}   camera: exposure {cam.get('exposureMs')} ms, "
                     f"gain {cam.get('gainDb')} dB ({cam.get('note', '')})"):
            fig.text(0.06, y, line, fontsize=9, family="monospace", va="top")
            y -= 0.022 * (1 + line.count("\n"))
        y -= 0.01
        for st in status.get("steps", []):
            head = f"[{st.get('status', ''):>7}] {st.get('title', '')}" + (f"  ({st['seconds']} s)" if st.get("seconds") is not None else "")
            body = "\n".join(filter(None, [wrap(st.get("detail", ""), 100)] + [wrap("warning: " + w, 100) for w in st.get("warnings", [])]))
            color = {"failed": "tab:red", "warn": "tab:orange", "ok": "tab:green"}.get(st.get("status"), "black")
            fig.text(0.06, y, head, fontsize=9, family="monospace", va="top", color=color, weight="bold")
            y -= 0.018
            if body:
                fig.text(0.09, y, body, fontsize=7.5, family="monospace", va="top")
                y -= 0.0145 * (1 + body.count("\n")) + 0.004
            if y < 0.12:
                break
        if settings:
            fig.text(0.06, 0.08, wrap("settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()), 120),
                     fontsize=6.5, family="monospace", va="top", color="dimgray")
        pdf.savefig(fig)

        for i in range(0, len(figures), 2):
            fig = Figure(figsize=(8.27, 11.69))
            for j, f in enumerate(figures[i:i + 2]):
                img = cv2.imdecode(np.frombuffer(f["jpg"], np.uint8), cv2.IMREAD_COLOR)
                top = 0.97 - j * 0.49
                ax = fig.add_axes([0.06, top - 0.33, 0.88, 0.31])
                ax.imshow(img[..., ::-1])
                ax.set_axis_off()
                ok = {True: " - ok", False: " - FAILED"}.get(f.get("ok"), "")
                fig.text(0.06, top, f"{f.get('step', '')}: {f.get('title', '')}{ok}", fontsize=10, weight="bold", va="top",
                         color="tab:red" if f.get("ok") is False else "black")
                fig.text(0.06, top - 0.345, wrap("expected (yellow): " + f.get("expected", ""), 105), fontsize=8, va="top", color="#a07800")
                fig.text(0.06, top - 0.395, wrap("measured (green): " + f.get("measured", ""), 105), fontsize=8, va="top", color="darkgreen")
            pdf.savefig(fig)


def zip_dir(folder):
    """ZIP of every file in ``folder`` (the raw report: JSON, 16-bit PNGs, figures, PDF)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                z.write(p, arcname=os.path.join(os.path.basename(folder.rstrip("/")), name))
    return buf.getvalue()


def _demo():
    import tempfile
    rng = np.random.default_rng(0)
    raw = (8 + 40 * rng.random((2048, 3072))).astype(np.uint16)
    cv2.circle(raw, (1536, 1024), 200, 4000, -1)
    a = annotate(raw, [("circle", (1536, 1024), 200, MEASURED), ("circle", (1536, 1024), 240, EXPECTED, True),
                       ("cross", (1535.5, 1023.5), 40, EXPECTED), ("arrow", (100, 100), (300, 100), OTHER),
                       ("text", (60, 60), "+X", OTHER)])
    assert a.shape == (640, 960, 3) and a.dtype == np.uint8
    assert tuple(a[320, 480]) != (0, 0, 0)
    ov = annotate(np.zeros((480, 640, 3), np.uint8), [("circle", (320, 240), 20, MEASURED)])
    assert ov.shape == (720, 960, 3)
    m = montage([a, ov, a])
    assert m.shape[1] == 960 and m.shape[0] >= 2 * 320
    curve = plot_curves([([0, 1, 2, 3], [1, 3, 2, 1], "coarse")], "Z (µm)", "focus metric", mark_x=1)
    assert curve is not None and curve.shape[1] == 960
    status = {"state": "failed", "message": "First objective: lost the via", "startedAt": "t0", "finishedAt": "t1",
              "steps": [{"name": "preflight", "title": "Preflight", "status": "ok", "detail": "WidefieldCamera 3072x2048",
                         "warnings": ["dark"], "seconds": 0.5}], "result": None}
    figs = [{"step": "objective0", "title": "via", "expected": "disc ⌀ 486 px", "measured": "⌀ 412 px, 0.598 µm/px",
             "ok": True, "jpg": jpeg(a)}, {"step": "focus", "title": "curve", "expected": "one peak", "measured": "Z -4380",
                                          "ok": False, "jpg": jpeg(curve)}, {"step": "x", "title": "ov", "expected": "",
                                                                             "measured": "", "ok": None, "jpg": jpeg(ov)}]
    with tempfile.TemporaryDirectory() as d:
        write_pdf(os.path.join(d, "report.pdf"), status, {"illuminationValue": 1023}, figs)
        assert open(os.path.join(d, "report.pdf"), "rb").read(5) == b"%PDF-"
        z = zip_dir(d)
        assert zipfile.ZipFile(io.BytesIO(z)).namelist()[0].endswith("report.pdf")
    print("frame_calibration_report self-check OK")


if __name__ == "__main__":
    _demo()
