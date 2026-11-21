"""Lightsheet volumes on disk: listing, a small preview for the browser 3D view, tile layout.

Pure functions (no controller state), so they can be tested without a microscope.
"""
import json
import math
import os
import re
import struct
from contextlib import contextmanager

import numpy as np
import tifffile
import zarr

TIFF_SUFFIXES = (".ome.tif", ".ome.tiff", ".tif", ".tiff")


def resolve_data_path(data_dir: str, rel: str) -> str:
    """Absolute path of ``rel`` (relative to the data folder); refuses paths outside it."""
    root = os.path.realpath(data_dir)
    path = os.path.realpath(os.path.join(root, rel.replace("\\", "/").lstrip("/")))
    if path != root and not path.startswith(root + os.sep):
        raise ValueError(f"{rel!r} is outside the data folder")
    if not os.path.exists(path):
        raise FileNotFoundError(rel)
    return path


def data_relative(data_dir: str, path: str) -> str:
    """``/LightSheet/...``: the form the frontend puts behind ``/imswitch/data``."""
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(data_dir))
    return "/" + rel.replace(os.sep, "/")


def list_volumes(folder: str, data_dir: str) -> list:
    """OME-TIFF / OME-Zarr volumes below ``folder``, newest first."""
    out = []
    if not os.path.isdir(folder):
        return out
    for dirpath, dirnames, filenames in os.walk(folder):
        stores = [d for d in dirnames if d.endswith(".zarr") and not d.startswith(".")]
        dirnames[:] = [d for d in dirnames if not d.endswith(".zarr") and not d.startswith(".")]
        tiffs = [f for f in filenames if f.lower().endswith(TIFF_SUFFIXES) and not f.startswith(".")]
        for name in stores + tiffs:
            path = os.path.join(dirpath, name)
            out.append({
                "name": name,
                "folder": os.path.basename(dirpath),
                "path": data_relative(data_dir, path),
                "absolutePath": path,
                "kind": "zarr" if name.endswith(".zarr") else "tiff",
                "mtime": os.path.getmtime(path),
                "sizeBytes": os.path.getsize(path) if name not in stores else None,
            })
    return sorted(out, key=lambda v: v["mtime"], reverse=True)


def _ome_pixels(tif: tifffile.TiffFile) -> dict:
    """Attributes of the first OME ``<Pixels>`` element (empty for plain TIFFs)."""
    match = re.search(r"<Pixels\b[^>]*>", tif.ome_metadata or "")
    return dict(re.findall(r'(\w+)="([^"]*)"', match.group(0))) if match else {}


class _TiffLevel:
    """One pyramid level of a TIFF series, sliceable like a zarr array: reads only the
    pages (planes) the selection touches. (tifffile's aszarr needs a newer tifffile
    than the one pinned next to zarr 3.2.)"""

    def __init__(self, level, axes: str):
        self.level, self.shape, self.dtype = level, tuple(level.shape), level.dtype
        self.iy = axes.index("y")

    def __getitem__(self, sel):
        lead = np.arange(int(np.prod(self.shape[:self.iy]))).reshape(self.shape[:self.iy])[sel[:self.iy]]
        lead = np.asarray(lead)
        planes = self.level.asarray(key=lead.ravel().tolist())
        return planes.reshape(lead.shape + self.shape[self.iy:])[(Ellipsis,) + tuple(sel[self.iy:])]


@contextmanager
def open_volume(path: str):
    """``(axes, arrays, scales)``: axes like "tczyx", one array per pyramid level
    (finest first, zarr-sliceable), and per level the µm per index along each axis."""
    if path.rstrip("/").endswith(".zarr"):
        group = zarr.open_group(path, mode="r")
        attrs = group.attrs.asdict()
        ms = attrs.get("ome", attrs)["multiscales"][0]          # NGFF 0.5 nests under "ome"
        axes = "".join(a["name"] if isinstance(a, dict) else a for a in ms["axes"]).lower()
        arrays, scales = [], []
        for d in ms["datasets"]:
            arrays.append(group[d["path"]])
            scales.append(next((t["scale"] for t in d.get("coordinateTransformations", [])
                                if t.get("type") == "scale"), [1.0] * len(axes)))
        yield axes, arrays, scales
        return
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.lower()
        if "z" not in axes:                    # plain multi-page TIFF: pages are the stack
            axes = re.sub("[iq]", "z", axes, count=1)
        px = _ome_pixels(tif)
        base = {"z": float(px.get("PhysicalSizeZ", 1.0)), "y": float(px.get("PhysicalSizeY", 1.0)),
                "x": float(px.get("PhysicalSizeX", 1.0))}
        arrays = [_TiffLevel(level, axes) for level in series.levels]
        full = arrays[0].shape
        scales = [[base.get(ax, 1.0) * full[i] / a.shape[i] for i, ax in enumerate(axes)] for a in arrays]
        yield axes, arrays, scales


def volume_preview(path: str, max_xy: int = 256, max_z: int = 256, channel: int = 0):
    """``(header, uint8 volume (z, y, x))``: the first time point of one channel, read from
    the coarsest pyramid level that is still ``max_xy`` across, then strided down.
    Contrast is stretched between the 0.5 and 99.9 percentiles."""
    with open_volume(path) as (axes, arrays, scales):
        def size(a, ax):
            return a.shape[axes.index(ax)] if ax in axes else 1

        level = 0
        for i, a in enumerate(arrays):
            if max(size(a, "y"), size(a, "x")) >= max_xy:
                level = i
        arr = arrays[level]
        sxy = max(1, max(size(arr, "y"), size(arr, "x")) // max_xy)   # at most 2x max_xy across
        sz = max(1, math.ceil(size(arr, "z") / max_z))
        sel = tuple(slice(None, None, sz) if ax == "z" else slice(None, None, sxy) if ax in "yx"
                    else min(channel, n - 1) if ax == "c" else 0
                    for ax, n in zip(axes, arr.shape))
        vol = np.asarray(arr[sel], dtype=np.float32)
        if "z" not in axes:
            vol = vol[None]
        lo, hi = (float(v) for v in np.percentile(vol, (0.5, 99.9)))
        u8 = np.clip((vol - lo) * (255.0 / max(hi - lo, 1e-9)), 0, 255).astype(np.uint8)
        sc, sc0 = dict(zip(axes, scales[level])), dict(zip(axes, scales[0]))
        header = {
            "shape": list(u8.shape),
            "spacingUm": [sc.get("z", 1.0) * sz, sc.get("y", 1.0) * sxy, sc.get("x", 1.0) * sxy],
            "stride": [sz, sxy, sxy],
            "zStepUm": sc0.get("z", 1.0),
            "pixelSizeUm": sc0.get("x", 1.0),
            "fullShape": [size(arrays[0], "z"), size(arrays[0], "y"), size(arrays[0], "x")],
            "channels": size(arrays[0], "c"),
            "timepoints": size(arrays[0], "t"),
            "level": level,
            "levels": len(arrays),
            "dtype": str(arrays[0].dtype),
            "range": [lo, hi],
        }
    return header, u8


def pack_preview(header: dict, volume: np.ndarray) -> bytes:
    """``uint32 little-endian header length | JSON header | raw uint8 voxels``."""
    blob = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(blob)) + blob + np.ascontiguousarray(volume).tobytes()


def tile_layout(stage_xy, affine, tile_shape):
    """Canvas placement of tiles taken at stage positions ``stage_xy`` [(x, y), ...] µm.

    ``affine`` is the 2x2 part of PixelCalibration's matrix (µm per px; it maps the
    registering shift, i.e. minus the content displacement, to stage µm), so the field of
    view of a tile ΔS away from the first lies ``inv(A) @ ΔS`` px (col, row) further on.
    Returns ``([(row, col), ...] top-left corners, (height, width) of the canvas)``.
    """
    # ponytail: translation only; tiles are not rotated, so a camera turned a few degrees
    # against the stage leaves seams. Register in BigStitcher/Ashlar if that matters.
    xy = np.asarray(stage_xy, dtype=float).reshape(-1, 2)
    colrow = (np.linalg.inv(np.asarray(affine, dtype=float)) @ (xy - xy[0]).T).T
    colrow = np.rint(colrow - colrow.min(axis=0)).astype(int)
    h, w = tile_shape
    return ([(int(r), int(c)) for c, r in colrow],
            (int(colrow[:, 1].max() + h), int(colrow[:, 0].max() + w)))


def fov_um(affine, tile_shape):
    """Stage extent (x, y) in µm of one camera frame."""
    h, w = tile_shape
    fx, fy = np.abs(np.asarray(affine, dtype=float)) @ np.array([w, h], dtype=float)
    return float(fx), float(fy)
