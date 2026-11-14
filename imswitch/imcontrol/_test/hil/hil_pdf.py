"""Printable HIL results followed by the server's original calibration PDF."""
import base64
import io
import os
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether,
)

INK = colors.HexColor("#16324f")
MUTED = colors.HexColor("#526579")
STATUS = {"passed": "#187448", "failed": "#b42332", "error": "#a34a12", "skipped": "#526579"}


def write_pdf(folder, setup, results, photos, calibration, started):
    """Only an explicitly collected, current-run PDF may be appended."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleHIL", fontName="Helvetica-Bold", fontSize=25,
                              leading=30, textColor=INK, spaceAfter=15))
    styles.add(ParagraphStyle("SectionHIL", fontName="Helvetica-Bold", fontSize=15,
                              leading=20, textColor=INK, spaceBefore=14, spaceAfter=9,
                              keepWithNext=True))
    styles.add(ParagraphStyle("BodyHIL", fontSize=9, leading=13, textColor=INK,
                              spaceAfter=7, splitLongWords=True))
    styles.add(ParagraphStyle("SmallHIL", parent=styles["BodyHIL"], fontSize=8,
                              leading=11, textColor=MUTED))

    def p(text, style="BodyHIL"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#dce4ec"))
        canvas.line(42, 37, A4[0] - 42, 37)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(42, 24, "openUC2 | Hardware tests and FRAME calibration")
        canvas.drawRightString(A4[0] - 42, 24, f"Tests / {doc.page}")

    story = [p("Hardware tests\n& FRAME calibration", "TitleHIL")]
    totals = {state: sum(r["status"] == state for r in results.values()) for state in STATUS}
    outcome = "FAILED" if totals["failed"] or totals["error"] else "PASSED" if totals["passed"] else "NOT RUN"
    story += [p(f"TEST RESULT: {outcome}", "SectionHIL"),
              p("   |   ".join(f"{n} {state}" for state, n in totals.items()))]
    metadata = [("Started (UTC)", datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds")),
                ("Setup", setup), ("Tested target", os.environ.get("HIL_TARGET", "Not supplied")),
                ("Run", os.environ.get("HIL_RUN_URL", "Local run")),
                ("Calibration", calibration.get("state", "not started"))]
    table = Table([[p(k, "SmallHIL"), p(v)] for k, v in metadata], colWidths=[100, A4[0] - 184])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 9),
                              ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    story += [table, Spacer(1, 12), p("Calibration results are retained on the instrument. "
              "Deployment restoration does not roll back the setup. A failed calibration may have written partial changes."),
              p("The pre-calibration setup backup and full diagnostic output are included in the report artifacts.", "SmallHIL"),
              p("Test overview", "SectionHIL")]
    rows = [[p("Test", "SmallHIL"), p("Result", "SmallHIL"), p("Time", "SmallHIL")]]
    for name, r in results.items():
        rows.append([p(name, "SmallHIL"), Paragraph(f'<font color="{STATUS[r["status"]]}">{r["status"].upper()}</font>', styles["SmallHIL"]), p(f"{r['duration']:.1f}s", "SmallHIL")])
    table = Table(rows, colWidths=[A4[0] - 220, 80, 56], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2ebf3")),
                              ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#dce4ec")),
                              ("TOPPADDING", (0, 0), (-1, -1), 7),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(table)
    if results:
        story.append(PageBreak())
    for name, r in results.items():
        story += [p(name, "SectionHIL"), p(f"{r['status'].upper()}  |  {r['duration']:.1f} seconds")]
        if r["details"]:
            # Individual paragraphs let even a very long traceback paginate.
            story.extend(p(line, "SmallHIL") for line in r["details"].splitlines())
        # The calibration PDF contains its full-size figures; do not duplicate them here.
        if "test_frame_calibration.py" not in name:
            for label, data in photos.get(name, []):
                raw = io.BytesIO(base64.b64decode(data))
                width, height = ImageReader(raw).getSize()
                scale = min((A4[0] - 84) / width, 285 / height)
                story += [KeepTogether([p(label), Image(raw, width=width * scale, height=height * scale)]), Spacer(1, 12)]
    if not calibration.get("pdf") or calibration.get("error"):
        story += [PageBreak(), p("FRAME calibration", "TitleHIL"),
                  p(f"State: {calibration.get('state', 'not started')}", "SectionHIL"),
                  p(calibration.get("message", "Calibration was not selected or could not start. See test results."))]
        if calibration.get("error"):
            story += [p("Calibration / report collection error", "SectionHIL"), p(calibration["error"])]
        if not calibration.get("pdf"):
            story.append(p("No calibration PDF from this run is available. No historical report has been substituted."))
    front = folder / "test-report.pdf"
    SimpleDocTemplate(str(front), pagesize=A4, rightMargin=42, leftMargin=42,
                      topMargin=42, bottomMargin=50, title="Hardware tests and FRAME calibration").build(
                          story, onFirstPage=footer, onLaterPages=footer)
    with PdfWriter() as writer:
        writer.append(str(front), outline_item="Hardware tests")
        if calibration.get("pdf"):
            reader = PdfReader(calibration["pdf"])
            if not len(reader.pages):
                raise ValueError("Calibration PDF contains no pages")
            writer.append(reader, outline_item="FRAME calibration (original report)")
        writer.add_metadata({"/Title": "Hardware tests and FRAME calibration", "/Author": "openUC2"})
        target = folder / "hil-report.pdf"
        temporary = folder / "hil-report.pending.pdf"
        writer.write(str(temporary))
        temporary.replace(target)
