"""Tissue detection on a prescan: sections in, pen ink and glass edges out."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from imswitch.imcontrol.model import tissue_detection as td  # noqa: E402

UPP = 50.0
SLOT = {"minX": 0.0, "maxX": 27000.0, "minY": 0.0, "maxY": 77000.0}
FOV = (1830.0, 1220.0)
SECTION_A = (13500.0, 15000.0, 4000.0, 3000.0)     # x, y, radius x, radius y (µm)
SECTION_B = (9000.0, 40000.0, 3000.0, 5000.0)


def _px(x, y):
    return int(round(y / UPP)), int(round(x / UPP))


def _slide(colour=True):
    """A slot at 50 µm/px: holder, glass, label, two H&E sections, pen marks, coverslip end."""
    h, w = int(SLOT["maxY"] / UPP), int(SLOT["maxX"] / UPP)
    img = np.full((h, w, 3), 10, np.uint8)                       # holder: no light
    img[10:1530, 10:530] = (235, 235, 235)                       # glass
    img[1200:1530, 10:530] = (35, 35, 35)                        # printed label
    rng = np.random.default_rng(3)
    for cx, cy, rx, ry in (SECTION_A, SECTION_B):
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, _px(cx, cy)[::-1], (int(rx / UPP), int(ry / UPP)), 0, 0, 360, 1, -1)
        texture = rng.uniform(0.0, 1.0, (h, w))
        eosin = np.array([200, 130, 176], np.float32)
        haem = np.array([120, 100, 170], np.float32)
        tissue = eosin * (1 - texture[..., None] * 0.5) + haem * (texture[..., None] * 0.5)
        img[mask > 0] = tissue[mask > 0].astype(np.uint8)
    cv2.line(img, _px(5000, 52000)[::-1], _px(20000, 54000)[::-1], (60, 170, 80), 12)   # green pen
    cv2.circle(img, _px(20000, 30000)[::-1], int(800 / UPP), (25, 25, 30), -1)          # black dot
    img[int(57000 / UPP):int(57000 / UPP) + 2, 10:530] = (120, 120, 120)     # coverslip end
    if not colour:
        img = img.mean(axis=2).astype(np.uint8)
    return img


def _overview(colour=True):
    img = _slide(colour)
    h, w = img.shape[:2]
    strip = td.Strip(img, x=w * UPP / 2, y=h * UPP / 2, width_um=w * UPP, height_um=h * UPP)
    return td.compose([strip], upp=UPP, flatten=False)


def _inside(bounds, x, y):
    return bounds["minX"] <= x <= bounds["maxX"] and bounds["minY"] <= y <= bounds["maxY"]


def _covered(tiles, x, y):
    return any(abs(t["x"] - x) <= FOV[0] / 2 and abs(t["y"] - y) <= FOV[1] / 2 for t in tiles)


def test_colour_prescan_finds_the_sections_and_not_the_pen_or_the_coverslip():
    res = td.detect_slide(_overview(True), SLOT, FOV, 0.1)
    assert len(res["regions"]) == 2, [r["bounds"] for r in res["regions"]]
    a, b = sorted(res["regions"], key=lambda r: r["bounds"]["minY"])
    assert _inside(a["bounds"], SECTION_A[0], SECTION_A[1])
    assert _inside(b["bounds"], SECTION_B[0], SECTION_B[1])
    # area close to the ellipses' (pi * rx * ry), within the morphology's slack
    assert a["areaMm2"] == pytest.approx(np.pi * 4.0 * 3.0, rel=0.2)
    # the pen stroke and the black dot are ink, reported but not scanned
    assert len(res["ink"]) >= 2
    for region in res["regions"]:
        assert not _inside(region["bounds"], 20000, 30000)          # black dot
        assert region["bounds"]["maxY"] < 51000 or region["bounds"]["minY"] > 55000
    # every tile plan covers its whole section, extreme points included
    for (cx, cy, rx, ry), region in ((SECTION_A, a), (SECTION_B, b)):
        for x, y in ((cx, cy), (cx - rx, cy), (cx + rx, cy), (cx, cy - ry), (cx, cy + ry)):
            assert _covered(region["tiles"], x, y), (x, y)
    assert res["tileCount"] == len(a["tiles"]) + len(b["tiles"])


def test_blank_glass_spot_is_on_clear_glass():
    res = td.detect_slide(_overview(True), SLOT, FOV, 0.1)
    blank = res["blank"]
    assert blank is not None and blank["clearanceUm"] >= 1500
    x, y = blank["x"], blank["y"]
    assert 500 + 1500 <= x <= 26500 - 1500 and 500 + 1500 <= y <= 60000
    for cx, cy, rx, ry in (SECTION_A, SECTION_B):
        assert ((x - cx) / (rx + 1500)) ** 2 + ((y - cy) / (ry + 1500)) ** 2 > 1.0


def test_grey_prescan_still_finds_the_sections_and_drops_the_coverslip_end():
    res = td.detect_slide(_overview(False), SLOT, FOV, 0.1)
    found = [r for r in res["regions"]
             if _inside(r["bounds"], SECTION_A[0], SECTION_A[1]) or
             _inside(r["bounds"], SECTION_B[0], SECTION_B[1])]
    assert len(found) == 2
    for region in res["regions"]:   # nothing is a band across the slide at the coverslip end
        across = region["bounds"]["maxX"] - region["bounds"]["minX"]
        at_coverslip = region["bounds"]["minY"] <= 57000 <= region["bounds"]["maxY"]
        assert not (at_coverslip and across > 20000)
    assert res["ink"] == []          # grey cannot tell pen from stain


def test_empty_and_dark_slots_say_so():
    ov = _overview(True)
    ov.image[:] = 10
    res = td.detect_slide(ov, SLOT, FOV, 0.1)
    assert res["regions"] == [] and res["warnings"]
    blank = _overview(True)
    blank.image[:] = 0
    blank.image[10:1530, 10:530] = 235
    res = td.detect_slide(blank, SLOT, FOV, 0.1)
    assert res["regions"] == [] and any("No tissue" in w for w in res["warnings"])
    assert res["blank"] is not None


def test_tile_plan_is_a_snake_over_what_it_touches():
    ov = td.Overview(np.zeros((100, 100, 1), np.float32), np.ones((100, 100), bool), 0.0, 0.0, UPP)
    mask = np.zeros((100, 100), bool)
    mask[20:60, 30:90] = True                    # 3000 x 2000 µm
    tiles = td.plan_tiles(mask, ov, 0, 0, 1000.0, 1000.0, overlap=0.0)
    assert len(tiles) == 6
    assert [t["iX"] for t in tiles] == [0, 1, 2, 2, 1, 0]
    assert [t["iY"] for t in tiles] == [0, 0, 0, 1, 1, 1]
    xs = sorted({t["x"] for t in tiles})
    assert xs == pytest.approx([2000.0, 3000.0, 4000.0])


def test_compose_places_strips_in_stage_coordinates_and_corrects_their_vignetting():
    rng = np.random.default_rng(0)
    profile = np.linspace(0.7, 1.0, 20)[:, None, None] * np.ones((20, 200, 3))
    strips = []
    for i in range(6):
        img = (200 * profile * rng.uniform(0.98, 1.02, profile.shape)).astype(np.uint8)
        strips.append(td.Strip(img, x=5000.0, y=500.0 + 1000.0 * i,
                               width_um=10000.0, height_um=1000.0))
    ov = td.compose(strips, upp=UPP)
    assert ov.bounds["minX"] == pytest.approx(0.0) and ov.bounds["minY"] == pytest.approx(0.0)
    assert ov.image.shape[:2] == (120, 200)
    rows = ov.image[..., 0].mean(axis=1)
    assert rows.max() / rows.min() < 1.08     # the 30 % falloff across each strip is gone


def test_focus_points_lie_on_the_tissue_and_spread_out():
    res = td.detect_slide(_overview(True), SLOT, FOV, 0.1)
    for region, (cx, cy, rx, ry) in zip(sorted(res["regions"], key=lambda r: r["bounds"]["minY"]),
                                        (SECTION_A, SECTION_B)):
        points = region["focusPoints"]
        assert len(points) == td.focus_point_count(region["areaMm2"]) >= 4
        for p in points:   # inside the section, not on the glass around it
            assert ((p["x"] - cx) / rx) ** 2 + ((p["y"] - cy) / ry) ** 2 < 1.0
        xs, ys = [p["x"] for p in points], [p["y"] for p in points]
        assert max(xs) - min(xs) > rx and max(ys) - min(ys) > ry    # spread, not clustered


def test_scan_polygon_covers_the_section_with_centre_inside_tiling():
    """The plate map tiles a polygon with fields whose centre is inside it (from its
    corner, at the step); the scan polygon must make that cover the whole section."""
    overlap = 0.1
    res = td.detect_slide(_overview(True), SLOT, FOV, overlap)
    sx, sy = FOV[0] * (1 - overlap), FOV[1] * (1 - overlap)
    for region, (cx, cy, rx, ry) in zip(sorted(res["regions"], key=lambda r: r["bounds"]["minY"]),
                                        (SECTION_A, SECTION_B)):
        poly = np.array(region["scanPolygon"], np.float32)
        assert len(poly) >= 3
        x0, y0 = poly[:, 0].min(), poly[:, 1].min()
        nx = int((poly[:, 0].max() - x0) / sx) + 1
        ny = int((poly[:, 1].max() - y0) / sy) + 1
        centres = [(x0 + i * sx, y0 + j * sy) for j in range(ny) for i in range(nx)]
        tiles = [c for c in centres if cv2.pointPolygonTest(poly, c, False) >= 0]
        for t in np.linspace(0, 2 * np.pi, 24, endpoint=False):    # the section's rim
            x, y = cx + rx * np.cos(t), cy + ry * np.sin(t)
            assert any(abs(x - tx) <= FOV[0] / 2 and abs(y - ty) <= FOV[1] / 2 for tx, ty in tiles)


def test_touching_strips_leave_no_seam_on_the_overview():
    """Strips 1221 µm high, 1221 µm apart, on a 50 µm grid: no row stays empty."""
    strips = [td.Strip(np.full((10, 40, 3), 200, np.uint8), x=1000.0, y=610.5 + 1221.0 * i,
                       width_um=2000.0, height_um=1221.0) for i in range(12)]
    ov = td.compose(strips, upp=UPP, flatten=False)
    assert ov.covered.all()
    assert ov.image[..., 0].min() > 150


def test_navy_marker_is_ink_and_a_haematoxylin_blue_section_is_tissue():
    """Colours measured on a FRAME 4x prescan (OD R/G/B against glass 228/233/233):
    navy marker core 0.98/1.08/0.75 and its thin strokes 0.32/0.33/0.24,
    a haematoxylin-blue section 0.32/0.28/0.15."""
    glass = np.array([228.0, 233.0, 233.0])
    rgb = lambda od: tuple(int(v) for v in glass * 10 ** -np.array(od))   # noqa: E731
    h, w = int(SLOT["maxY"] / UPP), int(SLOT["maxX"] / UPP)
    img = np.full((h, w, 3), 10, np.uint8)
    img[10:1530, 10:530] = glass.astype(np.uint8)
    img[1200:1530, 10:530] = (35, 35, 35)
    cv2.ellipse(img, _px(13500, 20000)[::-1], (60, 50), 0, 0, 360, rgb((0.32, 0.28, 0.15)), -1)
    cv2.line(img, _px(5000, 35000)[::-1], _px(20000, 35000)[::-1], rgb((0.98, 1.08, 0.75)), 12)
    cv2.line(img, _px(5000, 42000)[::-1], _px(20000, 45000)[::-1], rgb((0.32, 0.33, 0.24)), 14)
    strip = td.Strip(img, x=w * UPP / 2, y=h * UPP / 2, width_um=w * UPP, height_um=h * UPP)
    res = td.detect_slide(td.compose([strip], upp=UPP, flatten=False), SLOT, FOV, 0.1)
    assert len(res["regions"]) == 1
    assert _inside(res["regions"][0]["bounds"], 13500, 20000)
    assert res["regions"][0]["bounds"]["maxY"] < 30000      # neither marker line joined it
    assert len(res["ink"]) >= 2


def _pale_section_slide():
    """A section stained on one side and pale on the other, a pale smear far
    away, dust on the glass and a black pigment spot inside the section."""
    h, w = int(SLOT["maxY"] / UPP), int(SLOT["maxX"] / UPP)
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), 10.0, np.float32)
    img[10:1530, 10:530] = 235.0 + rng.normal(0, 1.0, (1520, 520, 3))      # glass
    for _ in range(40):                                                    # dust specks
        r, c = int(rng.integers(30, 1500)), int(rng.integers(30, 510))
        img[r:r + 2, c:c + 2] = 150.0

    def ellipse(cx, cy, rx, ry):
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, _px(cx, cy)[::-1], (int(rx / UPP), int(ry / UPP)), 0, 0, 360, 1, -1)
        return mask > 0

    # pale: OD R/G/B 0.02/0.045/0.04 (the measured pale middle: 0.019/0.034/0.037)
    pale_rgb = 235.0 * 10.0 ** -np.array([0.02, 0.045, 0.04])
    img[ellipse(15500, 20000, 3500, 5000)] = pale_rgb
    img[ellipse(13500, 60000, 2000, 2000)] = pale_rgb                       # smear, far away
    stained = ellipse(9500, 20000, 3000, 5000)
    img[stained] = (200.0, 130.0, 176.0)                                   # eosin
    cv2.circle(img, _px(12000, 21000)[::-1], int(500 / UPP), (25, 25, 30), -1)   # pigment
    img = np.clip(img, 0, 255).astype(np.uint8)
    strip = td.Strip(img, x=w * UPP / 2, y=h * UPP / 2, width_um=w * UPP, height_um=h * UPP)
    return td.compose([strip], upp=UPP, flatten=False)


def test_the_pale_part_of_a_stained_section_is_scanned_and_a_pale_smear_is_not():
    res = td.detect_slide(_pale_section_slide(), SLOT, FOV, 0.1)
    assert len(res["regions"]) == 1, [r["bounds"] for r in res["regions"]]
    region = res["regions"][0]
    assert region["bounds"]["maxX"] > 17500                # the pale side, not only the stain
    assert region["bounds"]["maxY"] < 30000                # the smear at y 60 mm stays glass
    for x, y in ((9500, 20000), (15500, 20000), (18000, 20000), (12000, 21000)):
        assert _covered(region["tiles"], x, y), (x, y)
    # about the two ellipses (pi * (3.5 * 5 + 3 * 5) mm2 minus their overlap), within
    # the morphology's slack: no leak into the glass around them
    assert region["areaMm2"] < 1.2 * np.pi * (3.5 * 5 + 3.0 * 5)
    assert res["ink"] == []                                 # the pigment is in the section


def test_the_flat_field_spot_avoids_glass_that_is_brighter_than_the_rest():
    # a brighter patch where the glass is farthest from everything (as the
    # middle band of a FRAME slide was): a reference there tilted every tile
    h, w = int(SLOT["maxY"] / UPP), int(SLOT["maxX"] / UPP)
    img = np.full((h, w, 3), 10, np.uint8)
    img[10:1530, 10:530] = (228, 228, 228)
    img[380:780, 150:400] = (236, 236, 236)                       # +3.5 %: OD -0.015
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, _px(13500, 8000)[::-1], (60, 40), 0, 0, 360, 1, -1)
    img[mask > 0] = (200, 130, 176)
    strip = td.Strip(img, x=w * UPP / 2, y=h * UPP / 2, width_um=w * UPP, height_um=h * UPP)
    ov = td.compose([strip], upp=UPP, flatten=False)
    res = td.detect_slide(ov, SLOT, FOV, 0.1)
    blank = res["blank"]
    assert blank is not None
    # the field at the spot leaves the patch out (its outermost grid frames
    # may touch it: the median of nine frames ignores that)
    half_h, half_w = FOV[1] / 2 / UPP, FOV[0] / 2 / UPP
    row, col = _px(blank["x"], blank["y"])
    rows = max(0.0, min(row + half_h, 780) - max(row - half_h, 380))
    cols = max(0.0, min(col + half_w, 400) - max(col - half_w, 150))
    assert rows * cols < 0.05 * (4 * half_h * half_w), blank
    plain = td.detect_slide(ov, SLOT, FOV, 0.1, td.TissueParams(blank_od_tolerance=1.0,
                                                                 blank_od_spread=1.0))
    row, col = _px(plain["blank"]["x"], plain["blank"]["y"])
    assert 380 <= row <= 780 and 150 <= col <= 400              # where it went before
