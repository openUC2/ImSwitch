"""tile_registration on synthetic scans with known tile positions."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from imswitch.imcontrol.model import tile_registration as tr

RNG = np.random.default_rng(3)
# positive like camera data (brightness and flat-field assume it)
WORLD = 10 + gaussian_filter(RNG.normal(size=(1400, 1400)), 3) + 0.3 * RNG.normal(size=(1400, 1400))
T, UM_PX, STEP = 200, 2.0, 260.0          # tile px, µm/px, commanded step µm (35 % overlap)


def make_scan(nx=4, ny=3, jitter_um=40.0, flip_x=False):
    """Serpentine scan; true positions = commanded + jitter; returns tiles, commanded xy µm, truth px."""
    order = [(ix if iy % 2 == 0 else nx - 1 - ix, iy) for iy in range(ny) for ix in range(nx)]
    cmd = np.array([(ix * STEP, iy * STEP) for ix, iy in order])
    true = cmd + RNG.normal(0, jitter_um, cmd.shape)
    tiles = []
    for x, y in true:
        px, py = int(round(x / UM_PX)) + 100, int(round(y / UM_PX)) + 100
        t = WORLD[py:py + T, px:px + T]
        tiles.append(t[:, ::-1] if flip_x else t)   # flipped camera: image x opposes stage x
    return tiles, cmd, true


@pytest.mark.parametrize("flip_x", [False, True])
def test_recovers_positions(flip_x):
    tiles, cmd, true = make_scan(flip_x=flip_x)
    r = tr.analyse_tiles(tiles, cmd, UM_PX)
    meas = np.array(r["measured_um"])
    rel_true = (np.round(true / UM_PX) * UM_PX) - np.round(true[0] / UM_PX) * UM_PX
    rel_meas = meas - meas[0]
    assert r["summary"]["tiles_solved"] == len(tiles)
    assert np.max(np.abs(rel_meas - rel_true)) < 2 * UM_PX
    assert r["summary"]["image_sign_x"] == (-1 if flip_x else 1)
    assert r["summary"]["pair_rms_um"] < 1.0


def test_textureless_tiles_are_excluded():
    tiles, cmd, _ = make_scan()
    tiles[5] = np.full_like(tiles[5], tiles[5].mean())   # an opaque/blank tile
    r = tr.analyse_tiles(tiles, cmd, UM_PX)
    assert 5 not in r["solved"] and r["summary"]["tiles_usable"] == len(tiles) - 1
