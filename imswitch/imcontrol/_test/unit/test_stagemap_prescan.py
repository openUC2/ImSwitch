"""The prescan line layout: slots on the tile pitch, a strip that fits in memory."""

import numpy as np
import pytest

from imswitch.imcontrol.controller.controllers.StageMapController import prescanLayout


def test_slots_follow_the_tile_pitch_not_a_frame_cap():
    # A 10 mm line, 300 µm pitch → 34 exposures; the old cap of 400 was blind to the pitch.
    slots, stripW, stripH, sub, dx = prescanLayout(
        10_000, 300, 0.12, (2000, 3000), subsample=4, maxStripBytes=64 << 20)
    assert (slots, dx) == (34, 300)
    assert stripW == int(np.ceil(10_000 / (0.12 * 4)))
    assert stripH == 500 and sub == 4


def test_pitch_is_clamped_to_the_field_so_the_strip_has_no_gaps():
    _, _, _, _, dx = prescanLayout(5_000, 900, 0.12, (2000, 3000), 4, 64 << 20)
    assert dx == pytest.approx(3000 * 0.12)
    _, _, _, _, default = prescanLayout(5_000, 0, 0.12, (2000, 3000), 4, 64 << 20)
    assert default == pytest.approx(3000 * 0.12)


def test_strip_grows_coarser_until_it_fits():
    # 50 mm at 0.12 µm/px would be 416 k columns full-res; a 8 MB budget forces it down.
    slots, stripW, stripH, sub, _ = prescanLayout(
        50_000, 300, 0.12, (3000, 3000), subsample=2, maxStripBytes=8 << 20)
    assert sub > 2
    assert stripW * stripH * 2 <= 8 << 20
    # Slots never change with the strip scale: they are exposures, not pixels.
    assert slots == int(np.ceil(50_000 / 300))


def test_slot_columns_tile_the_strip_contiguously():
    slots, stripW, _, sub, dx = prescanLayout(1_000, 30, 0.5, (100, 100), 2, 64 << 20)
    scale = 0.5 * sub
    edges = [min(stripW, int(round(k * dx / scale))) for k in range(slots + 1)]
    assert edges[0] == 0 and edges[-1] == stripW
    assert all(b >= a for a, b in zip(edges, edges[1:]))
