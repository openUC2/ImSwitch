"""Colour prescan strips, their previews, and detectTissue on the map."""
import base64
import threading
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from imswitch.imcontrol.controller.controllers.StageMapController import (  # noqa: E402
    StageMapController, StageMapParams, TissueDetectionRequest, TissueSlide,
    estimateProfileShift, spreadBands)

QUIET = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None,
                              warning=lambda *a, **k: None, error=lambda *a, **k: None)


def _stub(**extra):
    stub = types.SimpleNamespace(_logger=QUIET, params=StageMapParams(), **extra)
    for name in ("_centreBand", "_makePreview", "_previewSize", "_decodePreview",
                 "_fieldOfViewUm", "_getPixelSizeUm"):
        setattr(stub, name, types.MethodType(getattr(StageMapController, name), stub))
    stub._previewTo8bit = StageMapController._previewTo8bit
    return stub


def test_centre_band_of_a_colour_frame_keeps_its_colour():
    frame = np.zeros((100, 400, 3), np.uint8)
    frame[..., 0], frame[..., 2] = 200, 50
    band = _stub()._centreBand(frame, widthPx=200, subsample=4)
    assert band.shape == (25, 50, 3)
    assert band[..., 0].min() == 200 and band[..., 2].max() == 50


def test_colour_bands_spread_into_a_colour_strip():
    bands = [(t, np.full((10, 4, 3), (10 * t, 100, 200), np.uint8)) for t in range(5)]
    strip = spreadBands(bands, 0.0, 4.0, span=40.0, scale=1.0)
    assert strip.shape == (10, 40, 3)
    assert strip[0, 0, 0] == 0 and strip[0, -1, 0] == 40   # first and last band at the ends
    assert (strip[..., 2] == 200).mean() > 0.4


def test_lag_estimate_on_colour_strips_matches_their_luminance():
    rng = np.random.default_rng(4)
    profile = rng.uniform(0, 255, 400)
    a = np.repeat(np.stack([np.roll(profile, 7)] * 3, axis=-1)[None], 5, axis=0)
    b = np.repeat(np.stack([profile] * 3, axis=-1)[None], 5, axis=0)
    assert estimateProfileShift(a, b) == estimateProfileShift(a.mean(axis=2), b.mean(axis=2)) == 7


def test_preview_of_an_rgb_frame_decodes_with_the_same_colours():
    stub = _stub()
    frame = np.zeros((60, 80, 3), np.uint8)
    frame[:, :40] = (220, 30, 30)     # red on the left
    frame[:, 40:] = (30, 30, 220)     # blue on the right
    preview = stub._makePreview(frame, size=(80, 60))
    rgb = stub._decodePreview(preview, colour=True)
    assert rgb.shape == (60, 80, 3)
    assert rgb[30, 10, 0] > 180 and rgb[30, 10, 2] < 80      # still red, not blue
    assert rgb[30, 70, 2] > 180 and rgb[30, 70, 0] < 80
    grey = stub._decodePreview(preview)
    assert grey.ndim == 2


def _section_strips(stub, upp=50.0):
    """A slot 27 x 77 mm with one pink section, cut into prescan strips as previews."""
    h, w = 1540, 540
    img = np.full((h, w, 3), 10, np.uint8)
    img[10:1530, 10:530] = 235
    cv2.ellipse(img, (270, 400), (80, 60), 0, 0, 360, (200, 130, 176), -1)
    tiles = []
    for i, y0 in enumerate(range(0, h, 154)):
        strip = img[y0:y0 + 154]
        tiles.append({"id": i, "kind": "prescan", "x": w * upp / 2,
                      "y": (y0 + strip.shape[0] / 2) * upp, "widthUm": w * upp,
                      "heightUm": strip.shape[0] * upp,
                      "preview": stub._makePreview(strip, size=(w, strip.shape[0]))})
    return tiles


def test_detect_tissue_finds_the_section_on_the_map():
    stub = _stub(_tilesLock=threading.Lock(), _prescanThread=None)
    frame = np.zeros((2048, 3072, 3), np.uint8)
    stub._detector = types.SimpleNamespace(getLatestFrame=lambda: frame, pixelSizeUm=[1.0, 0.6, 0.6])
    stub._tiles = _section_strips(stub)
    request = TissueDetectionRequest(slides=[TissueSlide(id="slot1", name="Slide 1", minX=0,
                                                         maxX=27000, minY=0, maxY=77000)])
    out = StageMapController.detectTissue(stub, request)
    assert out["success"] and out["colour"]
    assert out["fov"]["x"] == pytest.approx(3072 * 0.6)
    slide = out["slides"][0]
    assert slide["name"] == "Slide 1" and len(slide["regions"]) == 1
    region = slide["regions"][0]
    assert region["bounds"]["minX"] < 13500 < region["bounds"]["maxX"]
    assert region["bounds"]["minY"] < 20000 < region["bounds"]["maxY"]
    assert region["tiles"] and slide["blank"] is not None
    jpeg = base64.b64decode(out["overview"]["image"])
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (1540, 540)


def test_detect_tissue_without_a_prescan_says_what_to_do():
    stub = _stub(_tilesLock=threading.Lock(), _prescanThread=None, _tiles=[])
    out = StageMapController.detectTissue(stub, TissueDetectionRequest())
    assert not out["success"] and "prescan" in out["error"]
