#!/usr/bin/env python3
"""
Convert individual TIFF files saved by ImSwitch ExperimentController
into various viewer-friendly representations.

Filename convention produced by OMEWriter._write_individual_tiff:
    t{YYYYMMDD_HHMMSS}_x{X}_y{Y}_z{Z}_c{cIdx}_{channelName}_i{iter}_p{power}.tif

    X, Y, Z are in microns * 1000 (integer, sub-micron precision).
    i{iter} is a global sequential counter across all channels and positions.

Directory layout:
    <base_dir>/tiles/timepoint_XXXX/<filename>.tif

IMPORTANT: The Z coordinate in the filename reflects the actual focus position
at the time of capture (including sample tilt drift), so every XY tile has a
slightly different Z value.  This means Z cannot be used as a grouping key for
stitching.  Instead the script uses the JSON protocol file (if available) which
stores integer grid indices (iX, iY) per iterator, or falls back to clustering
the X/Y stage coordinates into grid indices automatically.

Outputs (selectable via --mode):
    composite   – Per-position Z/T composite stack for napari (multi-channel)
    stitch      – Per-channel XY stitched OME-TIFF for Fiji
    mip         – Max intensity projection per XY position, then stitched
    mip-composite – Per-channel MIP stitched, then merged as composite

Dependencies:
    pip install tifffile numpy

Usage examples:
    # Convert all timepoints, all modes
    python convert_experiment_tiffs.py /path/to/base_dir/tiles

    # Only stitch, auto-locate JSON protocol
    python convert_experiment_tiffs.py /path/to/base_dir/tiles --mode stitch

    # Explicit JSON path
    python convert_experiment_tiffs.py /path/to/base_dir/tiles --protocol /path/to/protocol.json --mode mip-composite

    # Specify output directory
    python convert_experiment_tiffs.py /path/to/base_dir/tiles -o /path/to/output
    
    python convert_experiment_tiffs.py     /Users/bene/Downloads/20260408_140128/

    
    python /Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/ImSwitch/scripts/convert_experiment_tiffs.py /Users/bene/ImSwitchConfig/data/ExperimentController/20260426_144145/20260426_144145_experiment0_0_experiment_0_/tiles/ --mode ashlar \
    --pixel-size 0.5 --maximum-shift 50 --align-channel 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tifffile as tif
except ImportError:
    sys.exit("tifffile is required: pip install tifffile")


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Pattern: t{date}_x{X}_y{Y}_z{Z}_c{cIdx}_{channelName}_i{iter}_p{power}.tif
_FILENAME_RE = re.compile(
    r"t(?P<timestamp>\d{8}_\d{6})"
    r"_x(?P<x>-?\d+)"
    r"_y(?P<y>-?\d+)"
    r"_z(?P<z>-?\d+)"
    r"_c(?P<c_idx>\d+)"
    r"_(?P<channel>[A-Za-z0-9_]+?)"
    r"_i(?P<iter>\d+)"
    r"_p(?P<power>\d+)"
    r"\.tif$"
)


@dataclass
class TileInfo:
    """Parsed metadata for a single TIFF tile."""
    filepath: str
    timestamp: str
    x: int          # microns * 1000
    y: int
    z: int
    c_idx: int
    channel: str
    iterator: int
    power: int
    timepoint: int = 0   # filled from directory name
    # Grid indices assigned from JSON protocol or coordinate clustering
    ix: int = -1
    iy: int = -1


def parse_filename(filepath: str) -> Optional[TileInfo]:
    """Parse an individual TIFF filename into a TileInfo."""
    basename = os.path.basename(filepath)
    m = _FILENAME_RE.match(basename)
    if m is None:
        return None
    return TileInfo(
        filepath=filepath,
        timestamp=m.group("timestamp"),
        x=int(m.group("x")),
        y=int(m.group("y")),
        z=int(m.group("z")),
        c_idx=int(m.group("c_idx")),
        channel=m.group("channel"),
        iterator=int(m.group("iter")),
        power=int(m.group("power")),
    )


# ---------------------------------------------------------------------------
# JSON protocol loader
# ---------------------------------------------------------------------------

def _find_protocol_json(tiles_dir: str) -> Optional[str]:
    """
    Auto-locate the experiment protocol JSON next to the tiles directory.
    Searches the parent and grandparent directories for a *_protocol.json file.
    """
    search_dirs = [
        os.path.dirname(tiles_dir),                # e.g. .../experiment0_0_.../
        os.path.dirname(os.path.dirname(tiles_dir)), # e.g. .../20260316_163004/
    ]
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith("_protocol.json"):
                return os.path.join(d, fname)
    return None


def load_protocol_grid(json_path: str) -> Dict[int, Tuple[int, int]]:
    """
    Load the snake_tiles list from the protocol JSON and return a mapping
    iterator → (iX, iY).  iX/iY are the integer grid column/row indices.
    """
    with open(json_path) as f:
        data = json.load(f)

    iterator_to_grid: Dict[int, Tuple[int, int]] = {}
    for row in data.get("snake_tiles", []):
        for entry in row:
            it = entry.get("iterator")
            ix = entry.get("iX")
            iy = entry.get("iY")
            if it is not None and ix is not None and iy is not None:
                iterator_to_grid[it] = (int(ix), int(iy))
    return iterator_to_grid


def _cluster_to_indices(values: List[int]) -> Dict[int, int]:
    """
    Map a list of raw coordinate values (µm*1000) to 0-based integer indices
    by sorting the unique values and assigning sequential indices.
    """
    unique = sorted(set(values))
    return {v: i for i, v in enumerate(unique)}


def assign_grid_indices(tiles: List[TileInfo], protocol_json: Optional[str]) -> None:
    """
    Assign ix/iy grid indices to every TileInfo in-place.

    Strategy A (preferred): use the JSON protocol file which maps iterator → (iX, iY).
      The tile filenames carry a frame-level iterator that advances by more than 1
      per XY position (e.g. one step per channel + autofocus frame).  We therefore
      group tiles by unique stage (x, y) coordinate, rank each group by its minimum
      tile iterator, and match that rank to the correspondingly ranked JSON entry
      (sorted by JSON iterator).  This is robust regardless of the per-frame
      stepping multiplier.

    Strategy B (fallback): cluster the raw X/Y stage coordinates into grid indices.
    """
    if protocol_json and os.path.isfile(protocol_json):
        print(f"  Using protocol JSON: {os.path.basename(protocol_json)}")
        iter_map = load_protocol_grid(protocol_json)

        # Group tiles by unique (x, y) stage position; record the minimum tile
        # iterator seen at each position so we can sort groups by scan order.
        xy_groups: Dict[Tuple[int, int], List[TileInfo]] = {}
        for tile in tiles:
            key = (tile.x, tile.y)
            xy_groups.setdefault(key, []).append(tile)

        # Sort XY groups by their minimum tile iterator → scan order rank
        ranked_xy = sorted(xy_groups.keys(),
                           key=lambda k: min(t.iterator for t in xy_groups[k]))

        # Sort JSON entries by their iterator value → scan order rank
        ranked_json = sorted(iter_map.keys())  # JSON iterators 0..N-1

        if len(ranked_xy) != len(ranked_json):
            print(f"  WARNING: {len(ranked_xy)} unique XY positions but "
                  f"{len(ranked_json)} JSON entries – using coordinate fallback")
            x_map = _cluster_to_indices([t.x for t in tiles])
            y_map = _cluster_to_indices([t.y for t in tiles])
            for tile in tiles:
                tile.ix = x_map[tile.x]
                tile.iy = y_map[tile.y]
            return

        # Build xy → (iX, iY) mapping by matching ranks
        xy_to_grid: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for xy_key, json_iter in zip(ranked_xy, ranked_json):
            xy_to_grid[xy_key] = iter_map[json_iter]

        for tile in tiles:
            ix, iy = xy_to_grid[(tile.x, tile.y)]
            tile.ix = ix
            tile.iy = iy

        # Shift all indices so minimum is 0
        min_ix = min(t.ix for t in tiles)
        min_iy = min(t.iy for t in tiles)
        for tile in tiles:
            tile.ix -= min_ix
            tile.iy -= min_iy

        print(f"  Grid indices assigned from JSON for {len(tiles)}/{len(tiles)} tiles "
              f"({len(ranked_xy)} unique XY positions)")
    else:
        # Fallback: derive grid indices from sorted unique X/Y coordinate values
        print("  No protocol JSON found – deriving grid indices from stage coordinates")
        x_map = _cluster_to_indices([t.x for t in tiles])
        y_map = _cluster_to_indices([t.y for t in tiles])
        for tile in tiles:
            tile.ix = x_map[tile.x]
            tile.iy = y_map[tile.y]


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def discover_tiles(tiles_dir: str, protocol_json: Optional[str] = None) -> List[TileInfo]:
    """Walk the tiles directory, parse all TIFF filenames, assign grid indices."""
    tiles: List[TileInfo] = []
    tiles_path = Path(tiles_dir)

    for tp_dir in sorted(tiles_path.iterdir()):
        if not tp_dir.is_dir():
            continue
        # Extract timepoint index from directory name (timepoint_XXXX)
        tp_match = re.match(r"timepoint_(\d+)", tp_dir.name)
        tp_idx = int(tp_match.group(1)) if tp_match else 0

        for tif_file in sorted(tp_dir.glob("*.tif")):
            info = parse_filename(str(tif_file))
            if info is not None:
                info.timepoint = tp_idx
                tiles.append(info)

    if not tiles:
        print(f"No matching TIFF files found under {tiles_dir}")
        return tiles

    print(f"Discovered {len(tiles)} tiles across "
          f"{len(set(t.timepoint for t in tiles))} timepoint(s)")

    # Auto-find JSON if not provided
    if protocol_json is None:
        protocol_json = _find_protocol_json(tiles_dir)

    assign_grid_indices(tiles, protocol_json)
    return tiles


# ---------------------------------------------------------------------------
# Multi-experiment (timelapse) tile discovery
# ---------------------------------------------------------------------------

# Matches e.g. "20260408_140128_experiment0_44_experiment_0_"
#                                              ^^ series index
_EXPERIMENT_DIR_RE = re.compile(r"_experiment\d+_(\d+)_")


def _is_multi_experiment_dir(base_dir: str) -> bool:
    """Return True if base_dir contains experiment-series subdirectories."""
    for entry in os.scandir(base_dir):
        if entry.is_dir() and _EXPERIMENT_DIR_RE.search(entry.name):
            return True
    return False


def discover_tiles_from_base_dir(
    base_dir: str,
    protocol_json: Optional[str] = None,
) -> List[TileInfo]:
    """
    Discover tiles from a session base directory that contains one
    experiment subdirectory per timepoint:

        <base_dir>/
            {date}_experiment0_{N}_experiment_{M}_/
                tiles/
                    timepoint_{XXXX}/
                        <filename>.tif

    The timelapse frame index is taken from N in the folder name.
    """
    tiles: List[TileInfo] = []
    base_path = Path(base_dir)

    experiment_dirs: List[Tuple[int, Path]] = []
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        m = _EXPERIMENT_DIR_RE.search(subdir.name)
        if m is None:
            continue
        experiment_dirs.append((int(m.group(1)), subdir))

    if not experiment_dirs:
        # Nothing matched — fall back to plain discover_tiles
        return discover_tiles(base_dir, protocol_json)

    for series_idx, exp_dir in sorted(experiment_dirs):
        tiles_subdir = exp_dir / "tiles"
        if not tiles_subdir.is_dir():
            continue
        for tp_dir in sorted(tiles_subdir.iterdir()):
            if not tp_dir.is_dir():
                continue
            for tif_file in sorted(tp_dir.glob("*.tif")):
                info = parse_filename(str(tif_file))
                if info is not None:
                    info.timepoint = series_idx
                    tiles.append(info)

    if not tiles:
        print(f"No matching TIFF files found under {base_dir}")
        return tiles

    print(
        f"Discovered {len(tiles)} tiles across "
        f"{len(set(t.timepoint for t in tiles))} timepoints "
        f"from {len(experiment_dirs)} experiment directories"
    )

    # No protocol JSON available for multi-experiment layouts —
    # derive grid indices from stage coordinates.
    assign_grid_indices(tiles, None)
    return tiles


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _unique_sorted(values):
    """Return sorted unique values from an iterable."""
    return sorted(set(values))


@dataclass
class ExperimentGrid:
    """Describes the full dimensionality of the experiment."""
    timepoints: List[int] = field(default_factory=list)
    ix_positions: List[int] = field(default_factory=list)   # grid column indices
    iy_positions: List[int] = field(default_factory=list)   # grid row indices
    channels: List[str] = field(default_factory=list)
    c_indices: List[int] = field(default_factory=list)

    # lookup: (timepoint, ix, iy, c_idx) → list of TileInfo (multiple Z per position)
    lookup: Dict[Tuple[int, int, int, int], List[TileInfo]] = field(default_factory=dict)

    @staticmethod
    def from_tiles(tiles: List[TileInfo]) -> "ExperimentGrid":
        grid = ExperimentGrid()
        grid.timepoints = _unique_sorted(t.timepoint for t in tiles)
        grid.ix_positions = _unique_sorted(t.ix for t in tiles)
        grid.iy_positions = _unique_sorted(t.iy for t in tiles)
        grid.channels = _unique_sorted(t.channel for t in tiles)
        grid.c_indices = _unique_sorted(t.c_idx for t in tiles)
        for t in tiles:
            key = (t.timepoint, t.ix, t.iy, t.c_idx)
            grid.lookup.setdefault(key, []).append(t)
        return grid

    def get_tiles(self, tp: int, ix: int, iy: int, c_idx: int) -> List[TileInfo]:
        """Return all tiles (Z stack) for a given grid position / channel / timepoint."""
        return self.lookup.get((tp, ix, iy, c_idx), [])

    def get_single(self, tp: int, ix: int, iy: int, c_idx: int) -> Optional[TileInfo]:
        """Return a single representative tile (first available) for a grid position."""
        tiles = self.get_tiles(tp, ix, iy, c_idx)
        return tiles[0] if tiles else None


def _read_tile(info: TileInfo) -> np.ndarray:
    """Read a TIFF tile and return the pixel array."""
    print(f"    Reading tile: {os.path.basename(info.filepath)}  "
          f"(tp={info.timepoint}, ix={info.ix}, iy={info.iy}, c_idx={info.c_idx}, "
          f"z={info.z}, channel='{info.channel}')")
    return tif.imread(info.filepath)


# ---------------------------------------------------------------------------
# Focus measure helper
# ---------------------------------------------------------------------------

def _is_rgb(frame: np.ndarray) -> bool:
    """Return True for colour images stored as (H, W, C) with C in {3, 4}."""
    return frame.ndim == 3 and frame.shape[2] in (3, 4)


_4GB = 4 * 1024 ** 3  # standard TIFF size limit


def _imwrite_auto(fpath: str, arr: np.ndarray, **kwargs):
    """
    Write a TIFF, automatically upgrading to BigTIFF when the array exceeds 4 GB.
    imagej=True and bigtiff=True are mutually exclusive in tifffile, so imagej
    is silently dropped when BigTIFF is required.
    """
    if arr.nbytes > _4GB:
        kwargs.pop("imagej", None)   # incompatible with bigtiff
        kwargs["bigtiff"] = True
    tif.imwrite(fpath, arr, **kwargs)


def _focus_measure(frame: np.ndarray) -> float:
    """
    Return normalized Laplacian variance as a focus measure score.

    Higher value → sharper image.
    Uses a 3×3 discrete Laplacian approximation without scipy dependency.
    For RGB/RGBA images the luminance channel is used.
    """
    if _is_rgb(frame):
        # Weighted luminance: ITU-R BT.601
        f = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(np.float32)
    else:
        f = frame.astype(np.float32)
    # Discrete Laplacian: 4*center - top - bottom - left - right
    lap = (4.0 * f[1:-1, 1:-1]
           - f[:-2, 1:-1]   # top
           - f[2:, 1:-1]    # bottom
           - f[1:-1, :-2]   # left
           - f[1:-1, 2:])   # right
    mean_intensity = float(np.mean(f)) + 1e-6
    return float(np.var(lap)) / (mean_intensity ** 2)


# ---------------------------------------------------------------------------
# Mode 1: Composite stack (per position, for napari)
# Same (ix, iy) over all z planes + time + channels → TCZYX stack
# ---------------------------------------------------------------------------

def build_composite_stacks(grid: ExperimentGrid, out_dir: str):
    """
    For every unique (ix, iy) grid position, build a TCZYX composite stack
    that napari can open directly as a multi-channel hyperstack.
    """
    print("\n=== Building composite stacks (napari) ===")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all (ix, iy) positions that have at least one tile
    xy_positions = sorted(set((t.ix, t.iy)
                              for tiles in grid.lookup.values()
                              for t in tiles))

    for pos_idx, (ix, iy) in enumerate(xy_positions):
        # Find the Z-stack depth from the actual tiles at this position.
        # Use the first available timepoint that has tiles for each channel,
        # rather than assuming everything exists at grid.timepoints[0].
        all_z_tiles = []
        for ci in grid.c_indices:
            for tp in grid.timepoints:
                tiles = grid.get_tiles(tp, ix, iy, ci)
                if tiles:
                    all_z_tiles.extend(tiles)
                    break
        if not all_z_tiles:
            continue
        nZ = max(len(grid.get_tiles(tp, ix, iy, ci))
                 for tp in grid.timepoints for ci in grid.c_indices)

        sample = _read_tile(all_z_tiles[0])
        h, w = sample.shape[:2]
        dtype = sample.dtype

        nT = len(grid.timepoints)
        nC = len(grid.c_indices)
        is_rgb = _is_rgb(sample)

        if is_rgb:
            # RGB composite: store as (T, Z, C, H, W, S) — one S=3 axis for colour.
            # Saved as TZCYXS so tifffile keeps colour intact.
            nS = sample.shape[2]
            stack = np.zeros((nT, nZ, nC, h, w, nS), dtype=dtype)
        else:
            # Greyscale: ImageJ TZCYX hyperstack
            stack = np.zeros((nT, nZ, nC, h, w), dtype=dtype)

        for it, tp in enumerate(grid.timepoints):
            for ic, ci in enumerate(grid.c_indices):
                z_tiles = sorted(grid.get_tiles(tp, ix, iy, ci),
                                 key=lambda t: t.z)
                for iz, tile in enumerate(z_tiles):
                    frame = _read_tile(tile)
                    stack[it, iz, ic] = frame[:h, :w]

        fname = f"composite_ix{ix:03d}_iy{iy:03d}.ome.tif"
        fpath = os.path.join(out_dir, fname)

        if is_rgb:
            metadata = {"axes": "TZCYXS", "Channel": {"Name": grid.channels}}
            _imwrite_auto(fpath, stack, photometric="rgb", metadata=metadata)
        else:
            metadata = {"axes": "TZCYX", "Channel": {"Name": grid.channels}}
            _imwrite_auto(fpath, stack, imagej=True, metadata=metadata)
        print(f"  [{pos_idx+1}/{len(xy_positions)}] {fname}  "
              f"shape={stack.shape}  dtype={dtype}")


# ---------------------------------------------------------------------------
# Mode 2: Stitched OME-TIFF (per channel, for Fiji)
# All XY positions in a grid for one channel → large canvas
# ---------------------------------------------------------------------------

def _compute_canvas_from_grid(grid: ExperimentGrid, h: int, w: int):
    """
    Compute canvas size and pixel offsets using iX/iY integer grid indices.
    Returns (canvas_h, canvas_w, offset_map) where offset_map maps
    (ix, iy) → (row_px, col_px).
    """
    nCols = len(grid.ix_positions)
    nRows = len(grid.iy_positions)
    canvas_h = nRows * h
    canvas_w = nCols * w

    # Map grid index value → sequential position (in case indices are non-contiguous)
    ix_seq = {v: i for i, v in enumerate(grid.ix_positions)}
    iy_seq = {v: i for i, v in enumerate(grid.iy_positions)}

    offset_map = {}
    for ix in grid.ix_positions:
        for iy in grid.iy_positions:
            col_px = ix_seq[ix] * w
            row_px = iy_seq[iy] * h
            offset_map[(ix, iy)] = (row_px, col_px)

    return canvas_h, canvas_w, offset_map


def _channel_c_idx(grid: ExperimentGrid, ch_name: str) -> Optional[int]:
    """Return the c_idx for a channel name, or None if not found."""
    for tile_list in grid.lookup.values():
        for t in tile_list:
            if t.channel == ch_name:
                return t.c_idx
    return None


def _mip_or_first(tiles: List[TileInfo]) -> np.ndarray:
    """Return MIP of a Z-stack.  Falls back to single frame if only one tile."""
    if len(tiles) == 1:
        return _read_tile(tiles[0])
    frames = [_read_tile(t) for t in tiles]
    return np.max(np.stack(frames, axis=0), axis=0)


def build_stitched_tiffs(grid: ExperimentGrid, out_dir: str):
    """
    For every channel × timepoint, stitch all XY grid positions onto a
    single canvas using MIP over Z per position.  Produces one output
    image per channel per timepoint.

    NOTE: Because different XY positions may have different absolute Z
    values (sample tilt / autofocus drift), there is no single Z-plane
    that spans all tiles.  MIP is used to collapse the Z stack at each
    position before compositing.
    """
    print("\n=== Building stitched OME-TIFFs (Fiji) ===")
    os.makedirs(out_dir, exist_ok=True)

    first_tile_list = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile_list[0])
    h, w = sample.shape[:2]
    dtype = sample.dtype
    is_rgb = _is_rgb(sample)

    canvas_h, canvas_w, offset_map = _compute_canvas_from_grid(grid, h, w)
    total = len(grid.channels) * len(grid.timepoints)
    count = 0

    for ch_name in grid.channels:
        ci = _channel_c_idx(grid, ch_name)
        if ci is None:
            continue

        for tp in grid.timepoints:
            count += 1
            if is_rgb:
                canvas = np.zeros((canvas_h, canvas_w, sample.shape[2]), dtype=dtype)
            else:
                canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)
            placed = 0

            for ix in grid.ix_positions:
                for iy in grid.iy_positions:
                    tiles = grid.get_tiles(tp, ix, iy, ci)
                    if not tiles:
                        continue
                    frame = _mip_or_first(sorted(tiles, key=lambda t: t.z))
                    row, col = offset_map[(ix, iy)]
                    fh, fw = frame.shape[:2]
                    canvas[row:row+fh, col:col+fw] = frame[:h, :w]
                    placed += 1

            if placed == 0:
                continue

            fname = f"stitched_{ch_name}_t{tp:04d}.ome.tif"
            fpath = os.path.join(out_dir, fname)
            if is_rgb:
                _imwrite_auto(fpath, canvas, photometric="rgb", compression="zlib")
            else:
                _imwrite_auto(fpath, canvas, compression="zlib")
            print(f"  [{count}/{total}] {fname}  "
                  f"canvas={canvas.shape}  tiles={placed}")


# ---------------------------------------------------------------------------
# Mode 3: MIP per XY → stitch
# For each (ix, iy, channel, timepoint) compute MIP over Z, then stitch
# ---------------------------------------------------------------------------

def _compute_mip(grid: ExperimentGrid, ix: int, iy: int,
                 c_idx: int, tp: int) -> Optional[np.ndarray]:
    """Compute max intensity projection over Z for a given grid position/channel/time."""
    tiles = grid.get_tiles(tp, ix, iy, c_idx)
    if not tiles:
        return None
    frames = [_read_tile(t) for t in sorted(tiles, key=lambda t: t.z)]
    return np.max(np.stack(frames, axis=0), axis=0)


def build_mip_stitched(grid: ExperimentGrid, out_dir: str):
    """
    For each channel × timepoint, compute per-position MIP over Z,
    then stitch into a single canvas.
    """
    print("\n=== Building MIP-stitched images (Fiji) ===")
    os.makedirs(out_dir, exist_ok=True)

    first_tile_list = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile_list[0])
    h, w = sample.shape[:2]
    dtype = sample.dtype
    is_rgb = _is_rgb(sample)

    canvas_h, canvas_w, offset_map = _compute_canvas_from_grid(grid, h, w)

    for ch_name in grid.channels:
        ci = _channel_c_idx(grid, ch_name)
        if ci is None:
            continue

        for tp in grid.timepoints:
            if is_rgb:
                canvas = np.zeros((canvas_h, canvas_w, sample.shape[2]), dtype=dtype)
            else:
                canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)
            placed = 0

            for ix in grid.ix_positions:
                for iy in grid.iy_positions:
                    mip = _compute_mip(grid, ix, iy, ci, tp)
                    if mip is None:
                        continue
                    row, col = offset_map[(ix, iy)]
                    fh, fw = mip.shape[:2]
                    canvas[row:row+fh, col:col+fw] = mip[:h, :w]
                    placed += 1

            if placed == 0:
                continue

            fname = f"mip_stitched_{ch_name}_t{tp:04d}.ome.tif"
            fpath = os.path.join(out_dir, fname)
            if is_rgb:
                _imwrite_auto(fpath, canvas, photometric="rgb", compression="zlib")
            else:
                _imwrite_auto(fpath, canvas, compression="zlib")
            print(f"  {fname}  canvas={canvas.shape}  tiles={placed}")


# ---------------------------------------------------------------------------
# Mode 4: MIP composite (merge channels into napari composite)
# Compute per-channel MIP-stitched canvases, then stack as composite TCZYX
# ---------------------------------------------------------------------------

def build_mip_composite(grid: ExperimentGrid, out_dir: str):
    """
    Compute per-channel MIP-stitched canvases and merge them into
    a multi-channel composite stack for napari.
    """
    print("\n=== Building MIP composite (napari) ===")
    os.makedirs(out_dir, exist_ok=True)

    first_tile_list = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile_list[0])
    h, w = sample.shape[:2]
    dtype = sample.dtype
    is_rgb = _is_rgb(sample)

    canvas_h, canvas_w, offset_map = _compute_canvas_from_grid(grid, h, w)

    nT = len(grid.timepoints)
    nC = len(grid.channels)

    if is_rgb:
        # RGB composite: TZCYXS; Z=1 since MIP collapses the Z axis
        nS = sample.shape[2]
        stack = np.zeros((nT, 1, nC, canvas_h, canvas_w, nS), dtype=dtype)
    else:
        # ImageJ hyperstacks: TZCYX; Z=1 since MIP collapses the Z axis
        stack = np.zeros((nT, 1, nC, canvas_h, canvas_w), dtype=dtype)

    for ic, ch_name in enumerate(grid.channels):
        ci = _channel_c_idx(grid, ch_name)
        if ci is None:
            continue

        for it, tp in enumerate(grid.timepoints):
            for ix in grid.ix_positions:
                for iy in grid.iy_positions:
                    mip = _compute_mip(grid, ix, iy, ci, tp)
                    if mip is None:
                        continue
                    row, col = offset_map[(ix, iy)]
                    fh, fw = mip.shape[:2]
                    stack[it, 0, ic, row:row+fh, col:col+fw] = mip[:h, :w]

    fname = "mip_composite.ome.tif"
    fpath = os.path.join(out_dir, fname)

    if is_rgb:
        metadata = {"axes": "TZCYXS", "Channel": {"Name": grid.channels}}
        _imwrite_auto(fpath, stack, photometric="rgb", metadata=metadata)
    else:
        metadata = {"axes": "TZCYX", "Channel": {"Name": grid.channels}}
        _imwrite_auto(fpath, stack, imagej=True, metadata=metadata)
    print(f"  {fname}  shape={stack.shape}  dtype={dtype}")


# ---------------------------------------------------------------------------
# Mode 5: Best-focus plane selection (post-processing autofocus)
# For each (x, y, channel, timepoint) score every Z-plane with a Laplacian-
# variance focus measure and keep only the sharpest plane.  The resulting
# "focused" image for each XY position is then stitched into a canvas.
# ---------------------------------------------------------------------------

def _best_focus_frame(grid: ExperimentGrid, ix: int, iy: int,
                      c_idx: int, tp: int) -> Optional[np.ndarray]:
    """
    Return the sharpest Z-plane for a given grid position/channel/timepoint.
    Uses normalised Laplacian variance as focus criterion.
    Returns None if no tiles are available.
    """
    best_frame = None
    best_score = -1.0
    for tile in sorted(grid.get_tiles(tp, ix, iy, c_idx), key=lambda t: t.z):
        frame = _read_tile(tile)
        score = _focus_measure(frame)
        if score > best_score:
            best_score = score
            best_frame = frame
    return best_frame


def build_best_focus_stitched(grid: ExperimentGrid, out_dir: str):
    """
    For each channel × timepoint, select the best-focus Z-plane at every
    XY position and stitch the result into a single canvas — similar to MIP
    but choosing the sharpest frame instead of the brightest projection.
    """
    print("\n=== Building best-focus stitched images (post-proc. autofocus) ===")
    os.makedirs(out_dir, exist_ok=True)

    first_tile_list = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile_list[0])
    h, w = sample.shape[:2]
    dtype = sample.dtype
    is_rgb = _is_rgb(sample)

    canvas_h, canvas_w, offset_map = _compute_canvas_from_grid(grid, h, w)

    for ch_name in grid.channels:
        ci = _channel_c_idx(grid, ch_name)
        if ci is None:
            continue

        for tp in grid.timepoints:
            if is_rgb:
                canvas = np.zeros((canvas_h, canvas_w, sample.shape[2]), dtype=dtype)
            else:
                canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)
            placed = 0

            for ix in grid.ix_positions:
                for iy in grid.iy_positions:
                    frame = _best_focus_frame(grid, ix, iy, ci, tp)
                    if frame is None:
                        continue
                    row, col = offset_map[(ix, iy)]
                    fh, fw = frame.shape[:2]
                    canvas[row:row+fh, col:col+fw] = frame[:h, :w]
                    placed += 1

            if placed == 0:
                continue

            fname = f"bestfocus_stitched_{ch_name}_t{tp:04d}.ome.tif"
            fpath = os.path.join(out_dir, fname)
            if is_rgb:
                _imwrite_auto(fpath, canvas, photometric="rgb", compression="zlib")
            else:
                _imwrite_auto(fpath, canvas, compression="zlib")
            print(f"  {fname}  canvas={canvas.shape}  tiles={placed}")


# ---------------------------------------------------------------------------
# Mode 6: Timelapse concatenation
# Same (ix, iy, channel) across all timepoints → T×C×Y×X hyperstack.
# Handles both single-position and multi-position grids:
#   - single position  → one T×C×Y×X file
#   - multi-position   → one T×C×Y_canvas×X_canvas stitched file
# ---------------------------------------------------------------------------

def _get_frame_for_timepoint(
    grid: ExperimentGrid, ix: int, iy: int, c_idx: int, tp: int, use_mip: bool
) -> Optional[np.ndarray]:
    """Return a single 2-D frame for the given (ix, iy, c_idx, tp) combination."""
    tiles = sorted(grid.get_tiles(tp, ix, iy, c_idx), key=lambda t: t.z)
    if not tiles:
        return None
    if use_mip and len(tiles) > 1:
        return np.max(np.stack([_read_tile(t) for t in tiles], axis=0), axis=0)
    # Single Z or no MIP requested: pick best-focus frame
    if len(tiles) == 1:
        return _read_tile(tiles[0])
    # Multiple Z without MIP → best-focus
    best, best_score = None, -1.0
    for t in tiles:
        f = _read_tile(t)
        s = _focus_measure(f)
        if s > best_score:
            best_score, best = s, f
    return best


def build_timelapse(grid: ExperimentGrid, out_dir: str, use_mip: bool = False):
    """
    Concatenate images from the same (ix, iy, channel) across all timepoints
    into a T×C×Y×X ImageJ/napari hyperstack.  For each unique (ix, iy)
    position a separate output file is written.

    When *use_mip* is True and multiple Z planes exist per (position, tp),
    they are MIP-projected before stacking.  Otherwise the best-focus plane
    (Laplacian variance) is selected.
    """
    label = "mip" if use_mip else "bestfocus"
    print(f"\n=== Building timelapse stacks (Z={label}) ===")
    os.makedirs(out_dir, exist_ok=True)

    xy_positions = sorted(
        set((t.ix, t.iy) for tiles in grid.lookup.values() for t in tiles)
    )

    # Find representative frame for shape / dtype
    def _find_sample() -> Optional[np.ndarray]:
        for tiles in grid.lookup.values():
            if tiles:
                return _read_tile(tiles[0])
        return None

    sample = _find_sample()
    if sample is None:
        print("  No tiles found – skipping")
        return
    h, w = sample.shape[:2]
    dtype = sample.dtype

    nT = len(grid.timepoints)
    nC = len(grid.c_indices)
    single_pos = len(xy_positions) == 1

    if single_pos:
        # Single XY position: output is T×C×Y×X directly.
        ix, iy = xy_positions[0]
        stack = np.zeros((nT, nC, h, w), dtype=dtype)
        for it, tp in enumerate(grid.timepoints):
            for ic, ci in enumerate(grid.c_indices):
                frame = _get_frame_for_timepoint(grid, ix, iy, ci, tp, use_mip)
                if frame is not None:
                    stack[it, ic] = frame[:h, :w]

        fname = f"timelapse_{label}.ome.tif"
        fpath = os.path.join(out_dir, fname)
        tif.imwrite(
            fpath, stack, imagej=True,
            metadata={"axes": "TCYX", "Channel": {"Name": grid.channels}},
        )
        print(f"  {fname}  shape={stack.shape}  dtype={dtype}  "
              f"({nT} timepoints, {nC} channels)")
    else:
        # Multi-position: produce one stitched T×C×Y_canvas×X_canvas file.
        _, _, offset_map = _compute_canvas_from_grid(grid, h, w)
        canvas_h = len(grid.iy_positions) * h
        canvas_w = len(grid.ix_positions) * w

        stack = np.zeros((nT, nC, canvas_h, canvas_w), dtype=dtype)
        for it, tp in enumerate(grid.timepoints):
            for ic, ci in enumerate(grid.c_indices):
                for ix, iy in xy_positions:
                    frame = _get_frame_for_timepoint(grid, ix, iy, ci, tp, use_mip)
                    if frame is None:
                        continue
                    row, col = offset_map[(ix, iy)]
                    fh, fw = frame.shape[:2]
                    stack[it, ic, row:row+fh, col:col+fw] = frame[:h, :w]

        fname = f"timelapse_stitched_{label}.ome.tif"
        fpath = os.path.join(out_dir, fname)
        tif.imwrite(
            fpath, stack, imagej=True,
            metadata={"axes": "TCYX", "Channel": {"Name": grid.channels}},
        )
        print(f"  {fname}  shape={stack.shape}  dtype={dtype}  "
              f"({nT} timepoints, {nC} channels, "
              f"{len(xy_positions)} positions)")
# Mode 7: Ashlar-based stitching with sub-pixel alignment
# ---------------------------------------------------------------------------

def build_ashlar_stitched(grid: ExperimentGrid, out_dir: str,
                          pixel_size: float = 1.0,
                          maximum_shift: float = 50.0,
                          align_channel: int = 0):
    """
    Stitch tiles using ASHLAR (Alignment by Simultaneous Harmonization of
    Layer/Adjacency Registration) for sub-pixel-accurate stitching.

    For each timepoint, all channels are stitched together using ashlar's
    EdgeAligner so inter-tile shifts are globally optimised.  The result is
    written as a pyramidal OME-TIFF per timepoint.

    Parameters
    ----------
    grid : ExperimentGrid
        Parsed tile grid.
    out_dir : str
        Output directory.
    pixel_size : float
        Physical pixel size in microns (used for position conversion).
    maximum_shift : float
        Maximum allowed per-tile corrective shift in microns (ashlar -m).
    align_channel : int
        Channel index used for alignment (ashlar -c).
    """
    try:
        from ashlarUC2.scripts.ashlar import process_images, build_imswitch_reader
    except ImportError:
        try:
            from ashlar.scripts.ashlar import process_images, build_imswitch_reader
        except ImportError:
            print("  ERROR: ashlarUC2 (or ashlar) is not installed. "
                  "Install with: pip install ashlarUC2")
            return

    print("\n=== Building ashlar-stitched OME-TIFFs ===")
    os.makedirs(out_dir, exist_ok=True)

    for tp in grid.timepoints:
        # Collect all tile file paths for this timepoint (all channels)
        tile_paths = []
        for key, tile_list in grid.lookup.items():
            key_tp = key[0]
            if key_tp != tp:
                continue
            # Use MIP representative tile when Z-stack exists
            best = sorted(tile_list, key=lambda t: t.z)[0] if tile_list else None
            if best is not None:
                tile_paths.append(best.filepath)

        if not tile_paths:
            continue

        # Deduplicate (same file could appear for different z-slices)
        tile_paths = sorted(set(tile_paths))

        out_file = os.path.join(out_dir, f"ashlar_stitched_t{tp:04d}.ome.tif")
        print(f"  Timepoint {tp}: {len(tile_paths)} tiles → {os.path.basename(out_file)}")

        reader = build_imswitch_reader(tile_paths, pixel_size=pixel_size)

        result = process_images(
            filepaths=[reader],
            output=out_file,
            align_channel=align_channel,
            flip_x=False,
            flip_y=False,
            flip_mosaic_x=False,
            flip_mosaic_y=False,
            output_channels=None,
            maximum_shift=maximum_shift,
            stitch_alpha=0.01,
            maximum_error=None,
            filter_sigma=0,
            pyramid=out_file.endswith(".ome.tif"),
            tile_size=1024,
            ffp=None,
            dfp=None,
            barrel_correction=0,
            plates=False,
            quiet=False,
        )
        if result and result != 0:
            print(f"  WARNING: ashlar returned non-zero status {result}")
        else:
            print(f"  Written: {out_file}")


# ---------------------------------------------------------------------------
# Fiji TileConfiguration.txt (for precise Grid/Collection stitching)
# ---------------------------------------------------------------------------

def write_tile_configuration(grid: ExperimentGrid, out_dir: str):
    """
    Write a Fiji-compatible TileConfiguration.txt that maps each
    individual TIFF to its approximate pixel position.  Users can
    feed this into Fiji → Plugins → Stitching → Grid/Collection
    for sub-pixel overlap registration.
    """
    print("\n=== Writing TileConfiguration.txt (Fiji) ===")
    os.makedirs(out_dir, exist_ok=True)

    first_tile_list = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile_list[0])
    h, w = sample.shape[:2]

    _, _, offset_map = _compute_canvas_from_grid(grid, h, w)

    for ch_name in grid.channels:
        ci = _channel_c_idx(grid, ch_name)
        if ci is None:
            continue

        for tp in grid.timepoints:
            fname = f"TileConfiguration_{ch_name}_t{tp:04d}.txt"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w") as f:
                f.write("# Define the number of dimensions we are working on\n")
                f.write("dim = 2\n\n")
                f.write("# Define the image coordinates\n")

                for ix in grid.ix_positions:
                    for iy in grid.iy_positions:
                        tiles = grid.get_tiles(tp, ix, iy, ci)
                        if not tiles:
                            continue
                        # Use first Z tile as representative for the position
                        info = sorted(tiles, key=lambda t: t.z)[0]
                        row, col = offset_map[(ix, iy)]
                        rel_path = os.path.relpath(info.filepath, out_dir)
                        f.write(f"{rel_path}; ; ({col}, {row})\n")

            print(f"  {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_MODES = ["composite", "stitch", "mip", "mip-composite", "focus", "tile-config",
             "timelapse", "timelapse-mip", "ashlar"]

'''
Explanation of modes:
- composite: For each (x, y) position, create a TCZYX stack that includes all timepoints, channels, and z-planes. This is ideal for napari which can handle multi-dimensional hyperstacks.
- stitch: For each channel and timepoint, stitch all XY tiles at a given Z plane into a single large OME-TIFF. This is suitable for Fiji's Grid/Collection plugin.
- mip: For each (x, y, channel, timepoint), compute a max intensity projection over Z, then stitch these MIPs into a single canvas. This gives a quick overview of the XY layout without Z information.
- mip-composite: Compute per-channel MIP-stitched canvases and merge them into a multi-channel composite stack for napari. This allows viewing the MIP of all channels together in napari.
- tile-config: Write a TileConfiguration.txt file for each channel and timepoint that maps each individual TIFF to its pixel position. This can be used with Fiji's Grid/Collection plugin
    for precise stitching with sub-pixel registration.

Sample usage:
    # Convert all timepoints, all modes
    python convert_experiment_tiffs.py /path/to/base_dir/tiles
    # Only composite stacks
    python convert_experiment_tiffs.py /path/to/base_dir/tiles --mode composite
    # Only MIP stitched
    python convert_experiment_tiffs.py /path/to/base_dir/tiles --mode mip
    # Specify output directory
    python convert_experiment_tiffs.py /path/to/base_dir/tiles -o /path/to/output

'''

def main():
    parser = argparse.ArgumentParser(
        description="Convert ImSwitch experiment TIFFs into viewer-friendly representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tiles_dir",
        help="Path to the tiles/ directory (contains timepoint_XXXX subfolders)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <tiles_dir>/../converted)",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=ALL_MODES + ["all"],
        default=["all"],
        help="Conversion mode(s) to run (default: all)",
    )
    parser.add_argument(
        "--protocol", "--json",
        default=None,
        metavar="JSON",
        help=("Path to the experiment protocol JSON file "
              "(auto-detected if omitted; contains iX/iY grid indices)"),
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        metavar="MICRONS",
        help="Physical pixel size in microns (used by ashlar mode, default: 1.0)",
    )
    parser.add_argument(
        "--maximum-shift",
        type=float,
        default=50.0,
        metavar="MICRONS",
        help="Maximum per-tile alignment shift in microns for ashlar (default: 50)",
    )
    parser.add_argument(
        "--align-channel",
        type=int,
        default=0,
        metavar="CHANNEL",
        help="Channel index used for ashlar alignment (default: 0)",
    )
    args = parser.parse_args()

    tiles_dir = os.path.abspath(args.tiles_dir)
    if not os.path.isdir(tiles_dir):
        sys.exit(f"Not a directory: {tiles_dir}")

    out_dir = args.output or os.path.join(tiles_dir, "converted")
    os.makedirs(out_dir, exist_ok=True)

    modes = set(args.mode)
    if "all" in modes:
        modes = set(ALL_MODES)

    # Auto-detect layout: base dir with experiment subdirs vs plain tiles dir
    if _is_multi_experiment_dir(tiles_dir):
        print(f"Detected multi-experiment timelapse layout under: {tiles_dir}")
        tiles = discover_tiles_from_base_dir(tiles_dir, protocol_json=args.protocol)
    else:
        tiles = discover_tiles(tiles_dir, protocol_json=args.protocol)
    if not tiles:
        sys.exit(1)

    grid = ExperimentGrid.from_tiles(tiles)

    # Count unique Z per position (for info display)
    n_z_per_pos = max(
        (len(v) for v in grid.lookup.values()), default=0
    )
    print(f"\nExperiment grid:")
    print(f"  Timepoints  : {len(grid.timepoints)}")
    print(f"  Grid columns: {len(grid.ix_positions)} (iX)")
    print(f"  Grid rows   : {len(grid.iy_positions)} (iY)")
    print(f"  Z per pos   : up to {n_z_per_pos}")
    print(f"  Channels    : {grid.channels}")

    # Run requested conversions
    if "composite" in modes:
        build_composite_stacks(grid, os.path.join(out_dir, "composite"))

    if "stitch" in modes:
        build_stitched_tiffs(grid, os.path.join(out_dir, "stitched"))

    if "mip" in modes:
        build_mip_stitched(grid, os.path.join(out_dir, "mip_stitched"))

    if "mip-composite" in modes:
        build_mip_composite(grid, os.path.join(out_dir, "mip_composite"))

    if "focus" in modes:
        build_best_focus_stitched(grid, os.path.join(out_dir, "best_focus"))

    if "tile-config" in modes:
        write_tile_configuration(grid, os.path.join(out_dir, "tile_config"))

    if "timelapse" in modes:
        build_timelapse(grid, os.path.join(out_dir, "timelapse"), use_mip=False)

    if "timelapse-mip" in modes:
        build_timelapse(grid, os.path.join(out_dir, "timelapse_mip"), use_mip=True)
        
    if "ashlar" in modes:
        build_ashlar_stitched(
            grid,
            os.path.join(out_dir, "ashlar"),
            pixel_size=args.pixel_size,
            maximum_shift=args.maximum_shift,
            align_channel=args.align_channel,
        )

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
