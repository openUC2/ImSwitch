#!/usr/bin/env python3
"""
Stitch ImSwitch tile images using multiview-stitcher.

Tile filename convention (produced by ImSwitch ExperimentController):
    t{YYYYMMDD_HHMMSS}_x{X}_y{Y}_z{Z}_c{cIdx}_{channelName}_i{iter}_p{power}.tif

    X, Y, Z are stage coordinates in microns x 1000 (sub-micron integer precision).

The script performs:
    1. Discover tiles from the tiles/ directory (supports timepoint subdirectories
       and multi-experiment timelapse layouts).
    2. Assign grid indices from the protocol JSON (or fall back to coordinate clustering).
    3. Select the best-focus Z plane per (position, channel) using Laplacian variance.
    4. Register overlapping tile pairs with multiview-stitcher (phase-correlation).
    5. Fuse all tiles into a single stitched OME-TIFF per timepoint.

Dependencies:
    pip install multiview-stitcher spatial-image tifffile numpy

Usage:
    python stitch_multiview_stitcher.py /path/to/tiles_dir --pixel-size 0.325
    python stitch_multiview_stitcher.py /path/to/tiles_dir --pixel-size 0.325 --align-channel 1
    python stitch_multiview_stitcher.py /path/to/tiles_dir --pixel-size 0.325 --no-registration

    # Explicit protocol JSON and output directory:
    python stitch_multiview_stitcher.py /path/to/tiles_dir \\
        --protocol /path/to/protocol.json \\
        --pixel-size 0.325 \\
        -o /path/to/output
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
# Filename parsing  (identical convention to convert_experiment_tiffs.py)
# ---------------------------------------------------------------------------

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
    filepath: str
    timestamp: str
    x: int          # µm × 1000
    y: int
    z: int
    c_idx: int
    channel: str
    iterator: int
    power: int
    timepoint: int = 0
    ix: int = -1    # grid column index
    iy: int = -1    # grid row index


def parse_filename(filepath: str) -> Optional[TileInfo]:
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
# Protocol JSON — grid-index assignment
# ---------------------------------------------------------------------------

def _find_protocol_json(tiles_dir: str) -> Optional[str]:
    search_dirs = [
        os.path.dirname(tiles_dir),
        os.path.dirname(os.path.dirname(tiles_dir)),
    ]
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith("_protocol.json"):
                return os.path.join(d, fname)
    return None


def load_protocol_grid(json_path: str) -> Dict[int, Tuple[int, int]]:
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
    unique = sorted(set(values))
    return {v: i for i, v in enumerate(unique)}


def assign_grid_indices(tiles: List[TileInfo], protocol_json: Optional[str]) -> None:
    if protocol_json and os.path.isfile(protocol_json):
        print(f"  Using protocol JSON: {os.path.basename(protocol_json)}")
        iter_map = load_protocol_grid(protocol_json)

        xy_groups: Dict[Tuple[int, int], List[TileInfo]] = {}
        for tile in tiles:
            key = (tile.x, tile.y)
            xy_groups.setdefault(key, []).append(tile)

        ranked_xy = sorted(xy_groups.keys(),
                           key=lambda k: min(t.iterator for t in xy_groups[k]))
        ranked_json = sorted(iter_map.keys())

        if len(ranked_xy) != len(ranked_json):
            print(f"  WARNING: {len(ranked_xy)} XY positions vs {len(ranked_json)} "
                  f"JSON entries — falling back to coordinate clustering")
            x_map = _cluster_to_indices([t.x for t in tiles])
            y_map = _cluster_to_indices([t.y for t in tiles])
            for tile in tiles:
                tile.ix = x_map[tile.x]
                tile.iy = y_map[tile.y]
            return

        xy_to_grid: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for xy_key, json_iter in zip(ranked_xy, ranked_json):
            xy_to_grid[xy_key] = iter_map[json_iter]

        for tile in tiles:
            ix, iy = xy_to_grid[(tile.x, tile.y)]
            tile.ix = ix
            tile.iy = iy

        min_ix = min(t.ix for t in tiles)
        min_iy = min(t.iy for t in tiles)
        for tile in tiles:
            tile.ix -= min_ix
            tile.iy -= min_iy

        print(f"  Grid indices from JSON: {len(tiles)} tiles, "
              f"{len(ranked_xy)} unique XY positions")
    else:
        print("  No protocol JSON — deriving grid indices from stage coordinates")
        x_map = _cluster_to_indices([t.x for t in tiles])
        y_map = _cluster_to_indices([t.y for t in tiles])
        for tile in tiles:
            tile.ix = x_map[tile.x]
            tile.iy = y_map[tile.y]


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def discover_tiles(tiles_dir: str, protocol_json: Optional[str] = None) -> List[TileInfo]:
    tiles: List[TileInfo] = []
    tiles_path = Path(tiles_dir)
    for tp_dir in sorted(tiles_path.iterdir()):
        if not tp_dir.is_dir():
            continue
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

    if protocol_json is None:
        protocol_json = _find_protocol_json(tiles_dir)

    assign_grid_indices(tiles, protocol_json)
    return tiles


_EXPERIMENT_DIR_RE = re.compile(r"_experiment\d+_(\d+)_")


def _is_multi_experiment_dir(base_dir: str) -> bool:
    for entry in os.scandir(base_dir):
        if entry.is_dir() and _EXPERIMENT_DIR_RE.search(entry.name):
            return True
    return False


def discover_tiles_from_base_dir(
    base_dir: str,
    protocol_json: Optional[str] = None,
) -> List[TileInfo]:
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

    print(f"Discovered {len(tiles)} tiles across "
          f"{len(set(t.timepoint for t in tiles))} timepoints "
          f"from {len(experiment_dirs)} experiment directories")

    assign_grid_indices(tiles, None)
    return tiles


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def _unique_sorted(values):
    return sorted(set(values))


@dataclass
class ExperimentGrid:
    timepoints: List[int] = field(default_factory=list)
    ix_positions: List[int] = field(default_factory=list)
    iy_positions: List[int] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    c_indices: List[int] = field(default_factory=list)
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
        return self.lookup.get((tp, ix, iy, c_idx), [])


# ---------------------------------------------------------------------------
# Focus measure + Z-selection helpers
# ---------------------------------------------------------------------------

def _is_rgb(frame: np.ndarray) -> bool:
    return frame.ndim == 3 and frame.shape[2] in (3, 4)


def _focus_measure(frame: np.ndarray) -> float:
    if _is_rgb(frame):
        f = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1]
             + 0.114 * frame[:, :, 2]).astype(np.float32)
    else:
        f = frame.astype(np.float32)
    lap = (4.0 * f[1:-1, 1:-1]
           - f[:-2, 1:-1] - f[2:, 1:-1]
           - f[1:-1, :-2] - f[1:-1, 2:])
    mean_intensity = float(np.mean(f)) + 1e-6
    return float(np.var(lap)) / (mean_intensity ** 2)


def _select_best_z(tiles: List[TileInfo]) -> np.ndarray:
    if len(tiles) == 1:
        return tif.imread(tiles[0].filepath)
    best_img, best_score = None, -1.0
    for t in sorted(tiles, key=lambda t: t.z):
        img = tif.imread(t.filepath)
        score = _focus_measure(img)
        if score > best_score:
            best_score, best_img = score, img
    return best_img


# ---------------------------------------------------------------------------
# Core stitching with multiview-stitcher
# ---------------------------------------------------------------------------

def _make_xaffine(pixel_size_um: float, y_um: float, x_um: float):
    """Build a (3, 3) homogeneous affine xarray for scale + translation."""
    from multiview_stitcher import param_utils
    M = np.array([
        [pixel_size_um, 0.0,           y_um],
        [0.0,           pixel_size_um, x_um],
        [0.0,           0.0,           1.0 ],
    ])
    return param_utils.affine_to_xaffine(M)


def _write_tiff_pyramid(
    fpath: str,
    data: np.ndarray,
    photometric: str = 'minisblack',
    compression: str = 'zlib',
    pixel_size_um: float = 1.0,
    tile_size: int = 512,
) -> None:
    """Write a tiled, multi-resolution pyramidal BigTIFF.

    Levels are halved until the shortest side fits within one tile.
    Each level is computed with a 2×2 box-filter average to avoid aliasing.
    Physical pixel size is stored as TIFF resolution tags so viewers can
    display correct scale bars.
    """
    is_rgb = photometric == 'rgb'
    h = data.shape[0] if is_rgb else data.shape[-2]
    w = data.shape[1] if is_rgb else data.shape[-1]
    n_levels = max(1, int(np.ceil(np.log2(max(h, w) / tile_size))) + 1)

    def _ds2(arr: np.ndarray) -> np.ndarray:
        """2×2 box-filter downsample; works for (H,W,C) and (...,H,W)."""
        if is_rgb:
            eh, ew = arr.shape[0] & ~1, arr.shape[1] & ~1
            a = arr[:eh, :ew]
            return (
                (a[0::2, 0::2].astype(np.uint32)
                 + a[0::2, 1::2].astype(np.uint32)
                 + a[1::2, 0::2].astype(np.uint32)
                 + a[1::2, 1::2].astype(np.uint32)) >> 2
            ).astype(arr.dtype)
        else:
            eh, ew = arr.shape[-2] & ~1, arr.shape[-1] & ~1
            a = arr[..., :eh, :ew]
            return (
                (a[..., 0::2, 0::2].astype(np.uint32)
                 + a[..., 0::2, 1::2].astype(np.uint32)
                 + a[..., 1::2, 0::2].astype(np.uint32)
                 + a[..., 1::2, 1::2].astype(np.uint32)) >> 2
            ).astype(arr.dtype)

    # pixels per centimetre (TIFF resolutionunit=3)
    px_per_cm = 1.0 / (pixel_size_um * 1e-4)

    write_opts = dict(
        tile=(tile_size, tile_size),
        photometric=photometric,
        compression=compression,
        resolution=(px_per_cm, px_per_cm),
        resolutionunit=3,
    )

    with tif.TiffWriter(fpath, bigtiff=True) as tw:
        tw.write(data, subifds=n_levels - 1, **write_opts)
        level = data
        for _ in range(n_levels - 1):
            level = _ds2(level)
            tw.write(level, subfiletype=1, **write_opts)


def build_multiview_stitched(
    grid: ExperimentGrid,
    out_dir: str,
    pixel_size: float = 1.0,
    align_channel: int = 0,
    no_registration: bool = False,
    output_chunksize: int = 512,
    flip_x: bool = False,
    flip_y: bool = False,
) -> None:
    """
    Stitch tiles for every timepoint using multiview-stitcher.

    Strategy
    --------
    For each timepoint we select the best-focus Z plane per (XY position, channel)
    using Laplacian-variance.  The selected frames are registered pairwise on the
    alignment channel, then the same registration transforms are applied to all
    channels before fusion.

    When *no_registration* is True the metadata affine (stage position only) is
    used directly for fusion, which is faster but requires accurate stage coordinates.

    Parameters
    ----------
    grid           : ExperimentGrid
    out_dir        : str – output directory
    pixel_size     : float – physical pixel size in µm/pixel
    align_channel  : int – local channel index (0-based) used for registration
    no_registration: bool – skip registration, use metadata positions only
    output_chunksize: int – dask chunk size for fusion output in pixels
    flip_x         : bool – mirror each tile left↔right (horizontal flip) before stitching
    flip_y         : bool – mirror each tile top↔bottom (vertical flip) before stitching
    """
    try:
        import spatial_image as si
        from multiview_stitcher import msi_utils, registration, fusion
    except ImportError as exc:
        sys.exit(
            f"multiview-stitcher and spatial-image are required:\n"
            f"  pip install multiview-stitcher spatial-image\n"
            f"Error: {exc}"
        )

    os.makedirs(out_dir, exist_ok=True)

    n_channels = len(grid.c_indices)
    reg_c_local = min(align_channel, n_channels - 1)

    print(f"\n=== multiview-stitcher: {len(grid.timepoints)} timepoint(s), "
          f"{n_channels} channel(s), pixel_size={pixel_size} µm ===")
    if flip_x or flip_y:
        print(f"  Tile flips: {'flip_x ' if flip_x else ''}{'flip_y' if flip_y else ''}")
    if no_registration:
        print("  Registration: SKIPPED (using metadata positions only)")
    else:
        print(f"  Registration channel: {reg_c_local} ({grid.channels[reg_c_local]})")

    for tp in grid.timepoints:
        print(f"\n--- Timepoint {tp} ---")

        # Build per-channel lists of msims (one entry per XY position).
        # channel_msims[c_local][pos_idx] = msim  (grayscale, used for registration)
        # color_msims[c_local][color][pos_idx]    (per RGB plane, used for fusion)
        channel_msims: List[List] = [[] for _ in range(n_channels)]
        is_rgb_mode: bool = False
        n_colors: int = 0
        color_msims: List[List[List]] = []
        # Track stage positions for the log (use first channel for reference)
        n_positions = 0
        ref_size: Optional[Tuple[int, int]] = None

        for ix in grid.ix_positions:
            for iy in grid.iy_positions:
                has_any = any(
                    grid.get_tiles(tp, ix, iy, ci) for ci in grid.c_indices
                )
                if not has_any:
                    continue

                # Get reference stage position from the first available channel.
                ref_tile = None
                for ci in grid.c_indices:
                    z_tiles = grid.get_tiles(tp, ix, iy, ci)
                    if z_tiles:
                        ref_tile = z_tiles[0]
                        break

                y_um = ref_tile.y / 1000.0
                x_um = ref_tile.x / 1000.0
                xaffine = _make_xaffine(pixel_size, y_um, x_um)

                for c_local, ci in enumerate(grid.c_indices):
                    z_tiles = grid.get_tiles(tp, ix, iy, ci)
                    if z_tiles:
                        frame = _select_best_z(sorted(z_tiles, key=lambda t: t.z))
                    else:
                        # Missing channel at this position: fill with zeros
                        if ref_size is None:
                            # Need a real frame first to know the shape
                            for ci2 in grid.c_indices:
                                zt2 = grid.get_tiles(tp, ix, iy, ci2)
                                if zt2:
                                    frame0 = _select_best_z(sorted(zt2, key=lambda t: t.z))
                                    ref_size = frame0.shape[:2]
                                    break
                        frame = np.zeros(ref_size, dtype=np.uint16) if ref_size else None
                        if frame is None:
                            continue

                    if ref_size is None:
                        ref_size = frame.shape[:2]

                    # Apply axis flips before stitching
                    if flip_x:
                        frame = np.flip(frame, axis=1)
                    if flip_y:
                        frame = np.flip(frame, axis=0)

                    # Convert RGB to grayscale for stitching; store as 2D
                    if _is_rgb(frame):
                        frame_2d = (0.299 * frame[:, :, 0]
                                    + 0.587 * frame[:, :, 1]
                                    + 0.114 * frame[:, :, 2]).astype(frame.dtype)
                    else:
                        frame_2d = frame

                    sim = si.to_spatial_image(frame_2d, dims=['y', 'x'])
                    msim = msi_utils.get_msim_from_sim(sim)
                    msi_utils.set_affine_transform(
                        msim, xaffine=xaffine, transform_key='affine_metadata'
                    )
                    channel_msims[c_local].append(msim)

                    # For RGB tiles: build per-color-plane msims used for fusion
                    if _is_rgb(frame):
                        if not is_rgb_mode:
                            is_rgb_mode = True
                            n_colors = frame.shape[2]
                            color_msims = [
                                [[] for _ in range(n_colors)]
                                for _ in range(n_channels)
                            ]
                        for col in range(n_colors):
                            sim_col = si.to_spatial_image(
                                np.asarray(frame[:, :, col]), dims=['y', 'x']
                            )
                            msim_col = msi_utils.get_msim_from_sim(sim_col)
                            msi_utils.set_affine_transform(
                                msim_col, xaffine=xaffine,
                                transform_key='affine_metadata',
                            )
                            color_msims[c_local][col].append(msim_col)

                n_positions += 1

        if n_positions == 0:
            print(f"  No tiles found for timepoint {tp}, skipping.")
            continue

        print(f"  {n_positions} tile positions × {n_channels} channel(s)")

        # ------------------------------------------------------------------
        # Registration (on alignment channel; transforms copied to others)
        # ------------------------------------------------------------------
        fuse_key = 'affine_metadata'

        if not no_registration and len(channel_msims[reg_c_local]) > 1:
            print(f"  Registering {len(channel_msims[reg_c_local])} tiles "
                  f"on channel {reg_c_local} ({grid.channels[reg_c_local]})...")
            try:
                registration.register(
                    channel_msims[reg_c_local],
                    transform_key='affine_metadata',
                    new_transform_key='affine_registered',
                )
                fuse_key = 'affine_registered'

                # Copy registered transforms to all other channels
                for c_local in range(n_channels):
                    if c_local == reg_c_local:
                        continue
                    for i, msim in enumerate(channel_msims[c_local]):
                        src_msim = channel_msims[reg_c_local][i]
                        xaffine_t = msi_utils.get_transform_from_msim(
                            src_msim, transform_key='affine_registered'
                        )
                        # Drop the t-dim that registration adds (shape 1,3,3 → 3,3)
                        xaffine_static = xaffine_t.isel(t=0).drop_vars('t')
                        msi_utils.set_affine_transform(
                            msim,
                            xaffine=xaffine_static,
                            transform_key='affine_registered',
                        )

                # Copy registered transforms to per-color msims
                if is_rgb_mode:
                    for c_local in range(n_channels):
                        for col in range(n_colors):
                            for i, msim_col in enumerate(color_msims[c_local][col]):
                                src = channel_msims[reg_c_local][i]
                                xaffine_t = msi_utils.get_transform_from_msim(
                                    src, transform_key='affine_registered'
                                )
                                msi_utils.set_affine_transform(
                                    msim_col,
                                    xaffine=xaffine_t.isel(t=0).drop_vars('t'),
                                    transform_key='affine_registered',
                                )

                print(f"  Registration done (fusing with '{fuse_key}')")

            except Exception as exc:
                print(f"  WARNING: registration failed ({exc}); "
                      f"falling back to metadata positions")
                fuse_key = 'affine_metadata'
        else:
            if no_registration:
                pass  # intentional
            else:
                print("  Single tile — skipping registration")

        # ------------------------------------------------------------------
        # Fusion: per channel, then stack into (C, H, W)
        # ------------------------------------------------------------------
        print(f"  Fusing {n_channels} channel(s)...")
        fused_channels: List[np.ndarray] = []

        for c_local, ch_name in enumerate(grid.channels):
            if not channel_msims[c_local]:
                print(f"    Channel {c_local} ({ch_name}): no tiles, filling zeros")
                if fused_channels:
                    fused_channels.append(np.zeros_like(fused_channels[0]))
                continue

            try:
                if is_rgb_mode:
                    color_arrs = []
                    for col in range(n_colors):
                        fused_dt = fusion.fuse(
                            color_msims[c_local][col],
                            transform_key=fuse_key,
                            output_chunksize=output_chunksize,
                        )
                        fused_sim = msi_utils.get_sim_from_msim(fused_dt)
                        color_arrs.append(np.squeeze(np.array(fused_sim.values)))
                    arr = np.stack(color_arrs, axis=-1)  # (H, W, 3)
                    print(f"    Channel {c_local} ({ch_name}): RGB {arr.shape} {arr.dtype}")
                else:
                    fused_dt = fusion.fuse(
                        channel_msims[c_local],
                        transform_key=fuse_key,
                        output_chunksize=output_chunksize,
                    )
                    fused_sim = msi_utils.get_sim_from_msim(fused_dt)
                    arr = np.squeeze(np.array(fused_sim.values))
                    print(f"    Channel {c_local} ({ch_name}): {arr.shape} {arr.dtype}")
                fused_channels.append(arr)

            except Exception as exc:
                print(f"    Channel {c_local} ({ch_name}): fusion failed — {exc}")
                import traceback
                traceback.print_exc()
                if fused_channels:
                    fused_channels.append(np.zeros_like(fused_channels[0]))

        if not fused_channels:
            print(f"  No fused channels for tp={tp}, skipping output.")
            continue

        # ------------------------------------------------------------------
        # Save output
        # ------------------------------------------------------------------
        fname = f"multiview_stitched_t{tp:04d}.ome.tif"
        fpath = os.path.join(out_dir, fname)

        if is_rgb_mode:
            # fused_channels: list of (H, W, n_colors) arrays, one per microscope channel
            max_h = max(ch.shape[0] for ch in fused_channels)
            max_w = max(ch.shape[1] for ch in fused_channels)
            padded = []
            for ch in fused_channels:
                if ch.shape[0] != max_h or ch.shape[1] != max_w:
                    ch = np.pad(ch, ((0, max_h - ch.shape[0]),
                                     (0, max_w - ch.shape[1]),
                                     (0, 0)))
                padded.append(ch)
            result = padded[0] if len(padded) == 1 else np.stack(padded, axis=0)
            photometric = 'rgb'
        else:
            # Ensure all channels have the same shape (pad zeros if necessary)
            max_shape = tuple(
                max(ch.shape[i] for ch in fused_channels)
                for i in range(fused_channels[0].ndim)
            )
            padded = []
            for ch in fused_channels:
                if ch.shape != max_shape:
                    pad = [(0, max_shape[i] - ch.shape[i]) for i in range(ch.ndim)]
                    ch = np.pad(ch, pad)
                padded.append(ch)
            result = np.stack(padded, axis=0)  # (C, H, W)
            photometric = 'minisblack'

        _write_tiff_pyramid(
            fpath, result,
            photometric=photometric,
            compression='zlib',
            pixel_size_um=pixel_size,
        )
        print(f"  Written: {fname}  shape={result.shape}  dtype={result.dtype}")

    print(f"\nAll outputs in: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stitch ImSwitch tile images using multiview-stitcher "
            "(phase-correlation registration + weighted-average fusion)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tiles_dir",
        help="Path to the tiles/ directory (or base session directory for timelapse)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <tiles_dir>/multiview_stitched)",
    )
    parser.add_argument(
        "--protocol", "--json",
        default=None,
        metavar="JSON",
        help="Experiment protocol JSON (auto-detected if omitted; provides iX/iY indices)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        metavar="MICRONS",
        help="Physical pixel size in µm/pixel (required for accurate registration)",
    )
    parser.add_argument(
        "--align-channel",
        type=int,
        default=0,
        metavar="CHANNEL",
        help="Channel index (0-based) used for tile registration (default: 0)",
    )
    parser.add_argument(
        "--no-registration",
        action="store_true",
        default=False,
        help="Skip pairwise registration; fuse tiles using stage metadata positions only",
    )
    parser.add_argument(
        "--flip-x",
        action="store_true",
        default=False,
        help="Mirror each tile left↔right (horizontal flip) before stitching",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        default=False,
        help="Mirror each tile top↔bottom (vertical flip) before stitching",
    )
    parser.add_argument(
        "--output-chunksize",
        type=int,
        default=512,
        metavar="PIXELS",
        help="Dask chunk size for the fusion output (default: 512)",
    )

    args = parser.parse_args()

    tiles_dir = os.path.abspath(args.tiles_dir)
    if not os.path.isdir(tiles_dir):
        sys.exit(f"Not a directory: {tiles_dir}")

    out_dir = args.output or os.path.join(tiles_dir, "multiview_stitched")
    os.makedirs(out_dir, exist_ok=True)

    # Pixel size
    if args.pixel_size is not None:
        pixel_size = args.pixel_size
    else:
        pixel_size = 1.0
        print("WARNING: --pixel-size not provided; defaulting to 1.0 µm/pixel. "
              "Specify --pixel-size for accurate physical-space registration.")

    # Tile discovery
    if _is_multi_experiment_dir(tiles_dir):
        print(f"Detected multi-experiment timelapse layout under: {tiles_dir}")
        tiles = discover_tiles_from_base_dir(tiles_dir, protocol_json=args.protocol)
    else:
        tiles = discover_tiles(tiles_dir, protocol_json=args.protocol)

    if not tiles:
        sys.exit(1)

    grid = ExperimentGrid.from_tiles(tiles)

    n_z_per_pos = max((len(v) for v in grid.lookup.values()), default=0)
    print(f"\nExperiment grid:")
    print(f"  Timepoints  : {len(grid.timepoints)}")
    print(f"  Grid columns: {len(grid.ix_positions)} (iX)")
    print(f"  Grid rows   : {len(grid.iy_positions)} (iY)")
    print(f"  Z per pos   : up to {n_z_per_pos}")
    print(f"  Channels    : {grid.channels}")
    print(f"  Pixel size  : {pixel_size} µm")

    build_multiview_stitched(
        grid,
        out_dir,
        pixel_size=pixel_size,
        align_channel=args.align_channel,
        no_registration=args.no_registration,
        output_chunksize=args.output_chunksize,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )


if __name__ == "__main__":
    main()
