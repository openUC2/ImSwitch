"""hil-report.pdf: the results of the run, then the PDFs the tests appended.

The first pages are drawn here with ReportLab.
They show what was tested and one line per test file.
Then comes each test file with what failed, its notes, its cases and its photos.
pypdf appends the PDFs of ``hil_report.append_pdf`` as they are.
"""
import io
import os
import time
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Flowable, Image, Indenter, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import LayoutError

from hil_report import by_file, name_of, verdict_of

MARGIN = 42
PADDING = 6  # points ReportLab's page frame keeps free inside the margins, on each side
WIDTH = A4[0] - 2 * (MARGIN + PADDING)  # of the text
INDENT = 14                 # of a test's part under its heading
INK = colors.HexColor("#24515e")        # the headings
HEAD_ROW = colors.HexColor("#ececec")   # behind the column names of a table
COLOUR = {"passed": colors.HexColor("#2e9c3f"), "failed": colors.HexColor("#d0312d"),
          "error": colors.HexColor("#d0312d"), "skipped": colors.HexColor("#8a8a8a")}
OUTPUT_LINES = 12   # of a failure's traceback that go into the report
PHOTO_HEIGHT = 285  # points; a photo is scaled down to this

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=9, leading=11.5,
                      splitLongWords=True)
LABEL = ParagraphStyle("label", parent=BODY, fontName="Times-Bold")
CELLS = TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)])


def draw_verdict(canvas, x, y, radius, status):
    """A small square with round corners at (x, y): green with a tick, red with a cross,
    grey with a dash."""
    canvas.setFillColor(COLOUR[status])
    canvas.roundRect(x - radius, y - radius, 2 * radius, 2 * radius, radius * 0.35,
                     stroke=0, fill=1)
    canvas.setStrokeColor(colors.white)
    canvas.setLineWidth(radius * 0.3)
    d = radius * 0.45
    if status == "passed":
        canvas.line(x - d, y, x - d * 0.3, y - d)
        canvas.line(x - d * 0.3, y - d, x + d, y + d)
    elif status == "skipped":
        canvas.line(x - d, y, x + d, y)
    else:
        canvas.line(x - d, y - d, x + d, y + d)
        canvas.line(x - d, y + d, x + d, y - d)


class Verdict(Flowable):
    """The sign of draw_verdict, to put into a table."""

    def __init__(self, status, size=9):
        super().__init__()
        self.status, self.size = status, size

    def wrap(self, available_width, available_height):
        return self.size, self.size

    def draw(self):
        draw_verdict(self.canv, self.size / 2, self.size / 2, self.size / 2, self.status)


class Heading(Flowable):
    """A title on a thin line; with ``status``, its sign in front."""

    def __init__(self, text, size, status=None):
        super().__init__()
        self.text, self.size, self.status = text, size, status
        self.keepWithNext = True
        self.spaceBefore, self.spaceAfter = size, 4

    def wrap(self, available_width, available_height):
        self.width = available_width
        return available_width, self.size + 7

    def draw(self):
        x = 0
        if self.status:
            radius = self.size * 0.5
            draw_verdict(self.canv, radius, 5 + self.size * 0.35, radius, self.status)
            x = self.size + 5
        self.canv.setFillColor(INK)
        self.canv.setFont("Helvetica-Bold", self.size)
        self.canv.drawString(x, 5, self.text)
        self.canv.setStrokeColor(INK)
        self.canv.setLineWidth(0.5)
        self.canv.line(0, 0, self.width, 0)


def text(value, style=BODY):
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), style)


def rows(pairs, width):
    """A table of (label, value): the label bold on the left, as in the head of the report."""
    table = Table([[text(label, LABEL), value if isinstance(value, Flowable) else text(value)]
                   for label, value in pairs], colWidths=[90, width - 90], hAlign="LEFT")
    table.setStyle(CELLS)
    return table


def grid(head, body, widths):
    """A table with a line of column names, in bold on light grey."""
    table = Table([[text(name, LABEL) for name in head]] + body, colWidths=widths, repeatRows=1,
                  hAlign="LEFT")
    table.setStyle(CELLS)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEAD_ROW),
                               ("LEFTPADDING", (0, 0), (0, -1), 3)]))
    return table


def failure_lines(details):
    """The lines of a pytest traceback that say what went wrong: its "E   ..." lines."""
    lines = details.strip().splitlines()
    errors = [line[1:].strip() for line in lines if line.startswith("E ")]
    return (errors or lines)[-OUTPUT_LINES:]


def test_part(number, file, cases, description, notes, photos):
    """The flowables of one test file: its heading, what it checks, output, cases and photos."""
    name = name_of(file)
    width = WIDTH - INDENT
    output = []
    for case, result in cases.items():
        if result["status"] in ("failed", "error"):
            output.append(f"{case or name} {result['status']}:")
            output.extend(failure_lines(result["details"]))
        elif result["status"] == "skipped":
            output.append(f"{case} skipped: {result['details']}")
    output.extend(notes)
    pairs = [("What it checks", description)] if description else []
    # One line per row, labelled once: a table splits onto the next page only between rows.
    pairs += [("Output" if index == 0 else "", line) for index, line in enumerate(output)]
    part = [Heading(f"{number}. {name}", 11, verdict_of([r["status"] for r in cases.values()])),
            Indenter(left=INDENT)]
    # A file without docstring that passed has nothing to say.
    # ReportLab refuses a table without rows.
    if pairs:
        part.append(rows(pairs, width))
    if any(cases):
        body = [[text(case), text(result["status"]), text(f"{result['duration']:.1f} s")]
                for case, result in cases.items()]
        part += [Spacer(1, 4), text("Cases", LABEL),
                 grid(["Case", "Result", "Time"], body, [width - 130, 70, 60])]
    for case in cases:
        for label, png in photos.get(f"{file}::{case}", []):
            image_width, image_height = ImageReader(io.BytesIO(png)).getSize()
            scale = min(width / image_width, PHOTO_HEIGHT / image_height)
            part.append(KeepTogether([Spacer(1, 6), text(label), Image(
                io.BytesIO(png), width=image_width * scale, height=image_height * scale,
                hAlign="LEFT")]))
    return part + [Indenter(left=-INDENT)]


def test_pages(setup, started, results, descriptions, photos, notes):
    """The pages drawn here, as the bytes of a PDF."""
    files = by_file(results)
    statuses = [result["status"] for result in results.values()]
    counted = ", ".join(f"{statuses.count(status)} {status}"
                        for status in ("passed", "failed", "error", "skipped")
                        if status in statuses)
    head = [("Started", time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(started))),
            ("Duration", f"{(time.time() - started) / 60:.0f} min"),
            ("Result", counted or "no test ran"),
            ("Tested target", os.environ.get("HIL_TARGET")),
            ("Run", os.environ.get("HIL_RUN_URL"))]
    summary = []
    for number, (file, cases) in enumerate(files.items(), 1):
        summary.append([text(number), Verdict(verdict_of([r["status"] for r in cases.values()])),
                        text(name_of(file)), text(descriptions.get(file, "")),
                        text(f"{sum(r['duration'] for r in cases.values()):.0f} s")])
    story = [Heading(f"Hardware test: {setup}", 14, verdict_of(statuses)),
             rows([pair for pair in head if pair[1]], WIDTH),
             Spacer(1, 6), text("Summary", LABEL),
             grid(["No.", "Result", "Test", "What it checks", "Time"], summary,
                  [28, 42, 100, WIDTH - 215, 45])]
    # The rig (os-rpi) passes the images of the pallet under test, one per line.
    # One image per row: a table splits between its rows onto the next page, never inside a row.
    images = os.environ.get("HIL_IMAGES", "").split()
    if images:
        story += [Spacer(1, 6), text("Images under test", LABEL),
                  grid(["Image"], [[text(image)] for image in images], [WIDTH])]
    story.append(Heading("Tests", 13))
    for number, (file, cases) in enumerate(files.items(), 1):
        story += test_part(number, file, cases, descriptions.get(file, ""),
                           notes.get(file, []), photos)

    def page_number(canvas, document):
        canvas.setFont("Times-Roman", 8)
        canvas.drawRightString(A4[0] - MARGIN, 24, str(document.page))

    pages = io.BytesIO()
    SimpleDocTemplate(pages, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                      bottomMargin=MARGIN, title=f"Hardware test: {setup}").build(
                          story, onFirstPage=page_number, onLaterPages=page_number)
    return pages.getvalue()


def write_pdf(target, setup, started, results, descriptions, photos, notes, appended):
    """Write the report to the file ``target``.

    ``appended`` holds the (title, PDF bytes) the tests gave to ``hil_report.append_pdf``.
    """
    try:
        pages = test_pages(setup, started, results, descriptions, photos, notes)
    except LayoutError as error:
        raise ValueError(f"the report does not fit on its pages: {error}") from error
    with PdfWriter() as writer:
        writer.append(io.BytesIO(pages), outline_item="Hardware tests")
        for title, data in appended:
            writer.append(PdfReader(io.BytesIO(data)), outline_item=title)
        writer.add_metadata({"/Title": f"Hardware test: {setup}", "/Author": "openUC2"})
        # Written under another name first, so a run that dies here leaves no half report.
        pending = target.with_suffix(".pending")
        writer.write(str(pending))
        pending.replace(target)
