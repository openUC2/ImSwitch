#!/usr/bin/env python3
"""
Convert individual TIFF files saved by ImSwitch ExperimentController
into various viewer-friendly representations.

Filename convention produced by OMEWriter._write_individual_tiff:
    t{YYYYMMDD_HHMMSS}_x{X}_y{Y}_z{Z}_c{cIdx}_{channelName}_i{iter}_p{power}.tif

    X, Y, Z are in microns * 1000 (integer, sub-micron precision).

Directory layout:
    <base_dir>/tiles/timepoint_XXXX/<filename>.tif

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

    # Only composite stacks
    python convert_experiment_tiffs.py /path/to/base_dir/tiles --mode composite

    # Only MIP stitched
    python convert_experiment_tiffs.py /path/to/base_dir/tiles --mode mip

    # Specify output directory
    python convert_experiment_tiffs.py /path/to/base_dir/tiles -o /path/to/output
"""

from __future__ import annotations

import argparse
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
# Tile discovery
# ---------------------------------------------------------------------------

def discover_tiles(tiles_dir: str) -> List[TileInfo]:
    """Walk the tiles directory and parse all TIFF filenames."""
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
    else:
        print(f"Discovered {len(tiles)} tiles across "
              f"{len(set(t.timepoint for t in tiles))} timepoint(s)")
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
    x_positions: List[int] = field(default_factory=list)
    y_positions: List[int] = field(default_factory=list)
    z_positions: List[int] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    c_indices: List[int] = field(default_factory=list)

    # fast lookup: (t, x, y, z, c_idx) → TileInfo
    lookup: Dict[Tuple[int, int, int, int, int], TileInfo] = field(default_factory=dict)

    @staticmethod
    def from_tiles(tiles: List[TileInfo]) -> "ExperimentGrid":
        grid = ExperimentGrid()
        grid.timepoints = _unique_sorted(t.timepoint for t in tiles)
        grid.x_positions = _unique_sorted(t.x for t in tiles)
        grid.y_positions = _unique_sorted(t.y for t in tiles)
        grid.z_positions = _unique_sorted(t.z for t in tiles)
        grid.channels = _unique_sorted(t.channel for t in tiles)
        grid.c_indices = _unique_sorted(t.c_idx for t in tiles)
        for t in tiles:
            grid.lookup[(t.timepoint, t.x, t.y, t.z, t.c_idx)] = t
        return grid


def _read_tile(info: TileInfo) -> np.ndarray:
    """Read a TIFF tile and return the pixel array."""
    return tif.imread(info.filepath)


# ---------------------------------------------------------------------------
# Focus measure helper
# ---------------------------------------------------------------------------

def _focus_measure(frame: np.ndarray) -> float:
    """
    Return normalized Laplacian variance as a focus measure score.

    Higher value → sharper image.
    Uses a 3×3 discrete Laplacian approximation without scipy dependency.
    """
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
# Same (x, y) over all z planes + time + channels → TCZYX stack
# ---------------------------------------------------------------------------

def build_composite_stacks(grid: ExperimentGrid, out_dir: str):
    """
    For every unique (x, y) position, build a TCZYX composite stack
    that napari can open directly as a multi-channel hyperstack.
    """
    print("\n=== Building composite stacks (napari) ===")
    os.makedirs(out_dir, exist_ok=True)

    xy_positions = [(x, y) for x in grid.x_positions for y in grid.y_positions]
    # Filter to positions that actually have data
    xy_with_data = set((t.x, t.y) for key, t in grid.lookup.items())
    xy_positions = sorted(xy_with_data)

    for ix, (x, y) in enumerate(xy_positions):
        # Determine shape from first available tile
        sample_key = None
        for t in grid.timepoints:
            for z in grid.z_positions:
                for c in grid.c_indices:
                    if (t, x, y, z, c) in grid.lookup:
                        sample_key = (t, x, y, z, c)
                        break
                if sample_key:
                    break
            if sample_key:
                break
        if sample_key is None:
            continue

        sample = _read_tile(grid.lookup[sample_key])
        h, w = sample.shape[:2]
        dtype = sample.dtype

        nT = len(grid.timepoints)
        nC = len(grid.c_indices)
        nZ = len(grid.z_positions)

        # ImageJ hyperstacks require TZCYXS axis order
        stack = np.zeros((nT, nZ, nC, h, w), dtype=dtype)

        for it, tp in enumerate(grid.timepoints):
            for ic, ci in enumerate(grid.c_indices):
                for iz, zp in enumerate(grid.z_positions):
                    key = (tp, x, y, zp, ci)
                    if key in grid.lookup:
                        frame = _read_tile(grid.lookup[key])
                        stack[it, iz, ic] = frame[:h, :w]

        fname = f"composite_x{x}_y{y}.ome.tif"
        fpath = os.path.join(out_dir, fname)

        # Write as ImageJ-compatible hyperstack (axes must be TZCYX)
        metadata = {
            "axes": "TZCYX",
            "Channel": {"Name": grid.channels},
        }
        tif.imwrite(
            fpath, stack,
            imagej=True,
            metadata=metadata,
        )
        print(f"  [{ix+1}/{len(xy_positions)}] {fname}  "
              f"shape={stack.shape}  dtype={dtype}")


# ---------------------------------------------------------------------------
# Mode 2: Stitched OME-TIFF (per channel, for Fiji)
# All XY positions in a grid for one channel → large canvas
# ---------------------------------------------------------------------------

def _compute_canvas(grid: ExperimentGrid, h: int, w: int):
    """
    Compute canvas size and pixel offsets for XY positions.
    Returns (canvas_h, canvas_w, offset_map) where offset_map
    maps (x, y) → (row_px, col_px).
    """
    if len(grid.x_positions) < 2:
        px_step_x = w
    else:
        # Estimate tile spacing in pixels from coordinate differences
        dx = grid.x_positions[1] - grid.x_positions[0]
        px_step_x = max(abs(dx), 1)  # Will be rescaled below

    if len(grid.y_positions) < 2:
        px_step_y = h
    else:
        dy = grid.y_positions[1] - grid.y_positions[0]
        px_step_y = max(abs(dy), 1)

    # Map microns*1000 to pixel offsets.
    # The coordinate values are in µm*1000. We compute each tile's offset
    # relative to the minimum coordinate and scale so that the spacing
    # between adjacent tiles equals the tile width/height (simple grid layout,
    # no sub-pixel overlap).  If the user needs precise overlap stitching they
    # should use Fiji's Grid/Collection plugin on the individual TIFFs.
    x_min = grid.x_positions[0]
    y_min = grid.y_positions[0]

    # Simple grid: assign integer grid indices
    x_idx = {v: i for i, v in enumerate(grid.x_positions)}
    y_idx = {v: i for i, v in enumerate(grid.y_positions)}

    nRows = len(grid.y_positions)
    nCols = len(grid.x_positions)
    canvas_h = nRows * h
    canvas_w = nCols * w

    offset_map = {}
    for xv in grid.x_positions:
        for yv in grid.y_positions:
            col_px = x_idx[xv] * w
            row_px = y_idx[yv] * h
            offset_map[(xv, yv)] = (row_px, col_px)

    return canvas_h, canvas_w, offset_map


def build_stitched_tiffs(grid: ExperimentGrid, out_dir: str):
    """
    For every channel × timepoint × z-plane, stitch all XY tiles
    onto a single canvas and save as OME-TIFF (Fiji-friendly).
    """
    print("\n=== Building stitched OME-TIFFs (Fiji) ===")
    os.makedirs(out_dir, exist_ok=True)

    # Determine tile shape from first available tile
    first_tile = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile)
    h, w = sample.shape[:2]
    dtype = sample.dtype

    canvas_h, canvas_w, offset_map = _compute_canvas(grid, h, w)

    total = len(grid.channels) * len(grid.timepoints) * len(grid.z_positions)
    count = 0

    for ch_name in grid.channels:
        # Find the c_idx for this channel name
        ch_tiles = [t for t in grid.lookup.values() if t.channel == ch_name]
        if not ch_tiles:
            continue
        ci = ch_tiles[0].c_idx

        for tp in grid.timepoints:
            for zp in grid.z_positions:
                count += 1
                canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)

                placed = 0
                for xv in grid.x_positions:
                    for yv in grid.y_positions:
                        key = (tp, xv, yv, zp, ci)
                        if key not in grid.lookup:
                            continue
                        frame = _read_tile(grid.lookup[key])
                        row, col = offset_map[(xv, yv)]
                        fh, fw = frame.shape[:2]
                        canvas[row:row+fh, col:col+fw] = frame[:h, :w]
                        placed += 1

                if placed == 0:
                    continue

                fname = (f"stitched_{ch_name}"
                         f"_t{tp:04d}_z{zp}.ome.tif")
                fpath = os.path.join(out_dir, fname)
                tif.imwrite(fpath, canvas, compression="zlib")
                print(f"  [{count}/{total}] {fname}  "
                      f"canvas={canvas.shape}  tiles={placed}")


# ---------------------------------------------------------------------------
# Mode 3: MIP per XY → stitch
# For each (x, y, channel, timepoint) compute MIP over Z, then stitch
# ---------------------------------------------------------------------------

def _compute_mip(grid: ExperimentGrid, x: int, y: int,
                 c_idx: int, tp: int) -> Optional[np.ndarray]:
    """Compute max intensity projection over Z for a given position/channel/time."""
    frames = []
    for zp in grid.z_positions:
        key = (tp, x, y, zp, c_idx)
        if key in grid.lookup:
            frames.append(_read_tile(grid.lookup[key]))
    if not frames:
        return None
    return np.max(np.stack(frames, axis=0), axis=0)


def build_mip_stitched(grid: ExperimentGrid, out_dir: str):
    """
    For each channel × timepoint, compute per-position MIP over Z,
    then stitch into a single canvas.
    """
    print("\n=== Building MIP-stitched images (Fiji) ===")
    os.makedirs(out_dir, exist_ok=True)

    first_tile = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile)
    h, w = sample.shape[:2]
    dtype = sample.dtype

    canvas_h, canvas_w, offset_map = _compute_canvas(grid, h, w)

    for ch_name in grid.channels:
        ch_tiles = [t for t in grid.lookup.values() if t.channel == ch_name]
        if not ch_tiles:
            continue
        ci = ch_tiles[0].c_idx

        for tp in grid.timepoints:
            canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)
            placed = 0

            for xv in grid.x_positions:
                for yv in grid.y_positions:
                    mip = _compute_mip(grid, xv, yv, ci, tp)
                    if mip is None:
                        continue
                    row, col = offset_map[(xv, yv)]
                    fh, fw = mip.shape[:2]
                    canvas[row:row+fh, col:col+fw] = mip[:h, :w]
                    placed += 1

            if placed == 0:
                continue

            fname = f"mip_stitched_{ch_name}_t{tp:04d}.ome.tif"
            fpath = os.path.join(out_dir, fname)
            tif.imwrite(fpath, canvas, compression="zlib")
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

    first_tile = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile)
    h, w = sample.shape[:2]
    dtype = sample.dtype

    canvas_h, canvas_w, offset_map = _compute_canvas(grid, h, w)

    nT = len(grid.timepoints)
    nC = len(grid.channels)

    # ImageJ hyperstacks require TZCYXS axis order; Z=1 since it's a projection
    stack = np.zeros((nT, 1, nC, canvas_h, canvas_w), dtype=dtype)

    for ic, ch_name in enumerate(grid.channels):
        ch_tiles = [t for t in grid.lookup.values() if t.channel == ch_name]
        if not ch_tiles:
            continue
        ci = ch_tiles[0].c_idx

        for it, tp in enumerate(grid.timepoints):
            for xv in grid.x_positions:
                for yv in grid.y_positions:
                    mip = _compute_mip(grid, xv, yv, ci, tp)
                    if mip is None:
                        continue
                    row, col = offset_map[(xv, yv)]
                    fh, fw = mip.shape[:2]
                    stack[it, 0, ic, row:row+fh, col:col+fw] = mip[:h, :w]

    fname = "mip_composite.ome.tif"
    fpath = os.path.join(out_dir, fname)

    metadata = {
        "axes": "TZCYX",
        "Channel": {"Name": grid.channels},
    }
    tif.imwrite(fpath, stack, imagej=True, metadata=metadata)
    print(f"  {fname}  shape={stack.shape}  dtype={dtype}")


# ---------------------------------------------------------------------------
# Mode 5: Best-focus plane selection (post-processing autofocus)
# For each (x, y, channel, timepoint) score every Z-plane with a Laplacian-
# variance focus measure and keep only the sharpest plane.  The resulting
# "focused" image for each XY position is then stitched into a canvas.
# ---------------------------------------------------------------------------

def _best_focus_frame(grid: ExperimentGrid, x: int, y: int,
                      c_idx: int, tp: int) -> Optional[np.ndarray]:
    """
    Return the sharpest Z-plane for a given position/channel/timepoint.
    Uses normalised Laplacian variance as focus criterion.
    Returns None if no tiles are available.
    """
    best_frame = None
    best_score = -1.0
    for zp in grid.z_positions:
        key = (tp, x, y, zp, c_idx)
        if key not in grid.lookup:
            continue
        frame = _read_tile(grid.lookup[key])
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

    first_tile = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile)
    h, w = sample.shape[:2]
    dtype = sample.dtype

    canvas_h, canvas_w, offset_map = _compute_canvas(grid, h, w)

    for ch_name in grid.channels:
        ch_tiles = [t for t in grid.lookup.values() if t.channel == ch_name]
        if not ch_tiles:
            continue
        ci = ch_tiles[0].c_idx

        for tp in grid.timepoints:
            canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)
            placed = 0

            for xv in grid.x_positions:
                for yv in grid.y_positions:
                    frame = _best_focus_frame(grid, xv, yv, ci, tp)
                    if frame is None:
                        continue
                    row, col = offset_map[(xv, yv)]
                    fh, fw = frame.shape[:2]
                    canvas[row:row+fh, col:col+fw] = frame[:h, :w]
                    placed += 1

            if placed == 0:
                continue

            fname = f"bestfocus_stitched_{ch_name}_t{tp:04d}.ome.tif"
            fpath = os.path.join(out_dir, fname)
            tif.imwrite(fpath, canvas, compression="zlib")
            print(f"  {fname}  canvas={canvas.shape}  tiles={placed}")


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

    first_tile = next(iter(grid.lookup.values()))
    sample = _read_tile(first_tile)
    h, w = sample.shape[:2]

    _, _, offset_map = _compute_canvas(grid, h, w)

    for ch_name in grid.channels:
        ch_tiles = [t for t in grid.lookup.values() if t.channel == ch_name]
        if not ch_tiles:
            continue
        ci = ch_tiles[0].c_idx

        for tp in grid.timepoints:
            fname = f"TileConfiguration_{ch_name}_t{tp:04d}.txt"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w") as f:
                f.write("# Define the number of dimensions we are working on\n")
                f.write("dim = 2\n\n")
                f.write("# Define the image coordinates\n")

                for xv in grid.x_positions:
                    for yv in grid.y_positions:
                        key = (tp, xv, yv, grid.z_positions[0], ci)
                        if key not in grid.lookup:
                            continue
                        info = grid.lookup[key]
                        row, col = offset_map[(xv, yv)]
                        rel_path = os.path.relpath(info.filepath, out_dir)
                        f.write(f"{rel_path}; ; ({col}, {row})\n")

            print(f"  {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_MODES = ["composite", "stitch", "mip", "mip-composite", "focus", "tile-config"]

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
    args = parser.parse_args()

    tiles_dir = os.path.abspath(args.tiles_dir)
    if not os.path.isdir(tiles_dir):
        sys.exit(f"Not a directory: {tiles_dir}")

    out_dir = args.output or os.path.join(os.path.dirname(tiles_dir), "converted")
    os.makedirs(out_dir, exist_ok=True)

    modes = set(args.mode)
    if "all" in modes:
        modes = set(ALL_MODES)

    # Discover and parse tiles
    tiles = discover_tiles(tiles_dir)
    if not tiles:
        sys.exit(1)

    grid = ExperimentGrid.from_tiles(tiles)

    print(f"\nExperiment grid:")
    print(f"  Timepoints : {len(grid.timepoints)}")
    print(f"  XY positions: {len(grid.x_positions)} x {len(grid.y_positions)}")
    print(f"  Z planes   : {len(grid.z_positions)}")
    print(f"  Channels   : {grid.channels}")

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

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
