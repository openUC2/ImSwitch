"""
On-the-fly thumbnail generation + OME metadata extraction for the FileManager.

Thumbnails are generated lazily on first request and cached as JPEGs **outside** the
data folder (under the config dir), keyed by absolute path + mtime + size so they
invalidate automatically when a file changes. Works for OME-TIFF, OME-Zarr (directory
stores), plain TIFF, and standard images (PNG/JPG/BMP). No new dependencies — uses
``tifffile``, ``opencv-python`` (cv2), ``zarr`` and ``numpy``, all already required.

Public API:
    is_image(path)        -> (bool, image_type|None)   # image_type in {tiff, zarr, image}
    get_thumbnail(path, size=256) -> Path              # path to a cached JPEG
    extract_metadata(path) -> dict                     # OME/NGFF metadata as plain JSON
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Image extensions we can render. ".zarr" is handled separately (it is a *directory*).
TIFF_EXTS = (".ome.tif", ".ome.tiff", ".tif", ".tiff")
PLAIN_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
ZARR_SUFFIX = ".zarr"

_CACHE_CAP_BYTES = int(os.environ.get("IMSWITCH_THUMBNAIL_CACHE_MB", "512")) * 1024 * 1024
_JPEG_QUALITY = 85


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def is_image(path: str) -> Tuple[bool, Optional[str]]:
    """Return (is_image, image_type). ``*.zarr`` directories count as images."""
    p = str(path)
    low = p.lower()
    if low.endswith(ZARR_SUFFIX) and os.path.isdir(p):
        return True, "zarr"
    if os.path.isdir(p):
        return False, None
    if low.endswith(TIFF_EXTS):
        return True, "tiff"
    if low.endswith(PLAIN_IMAGE_EXTS):
        return True, "image"
    return False, None


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_dir() -> str:
    base = os.environ.get("IMSWITCH_THUMBNAIL_CACHE")
    if not base:
        cfg = None
        try:
            from imswitch.imcommon.model import dirtools
            cfg = getattr(dirtools.UserFileDirs, "Root", None)
        except Exception:
            cfg = None
        cfg = cfg or os.path.expanduser("~/ImSwitchConfig")
        base = os.path.join(cfg, ".cache", "thumbnails")
    os.makedirs(base, exist_ok=True)
    return base


def _cache_key(abs_path: str, size: int) -> str:
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = 0
    raw = f"{os.path.abspath(abs_path)}|{mtime}|{size}".encode("utf-8", "replace")
    return hashlib.sha1(raw).hexdigest()


def _prune_cache(cache_dir: str, cap_bytes: int = _CACHE_CAP_BYTES) -> None:
    """Evict oldest cached thumbnails once total size exceeds the cap (best-effort)."""
    try:
        entries = []
        total = 0
        for f in glob.glob(os.path.join(cache_dir, "*.jpg")):
            try:
                st = os.stat(f)
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, f))
            total += st.st_size
        if total <= cap_bytes:
            return
        entries.sort()  # oldest first
        for _mtime, sz, f in entries:
            try:
                os.remove(f)
                total -= sz
            except OSError:
                pass
            if total <= cap_bytes:
                break
    except Exception:
        pass  # cache pruning must never break a request


def get_thumbnail(abs_path: str, size: int = 256) -> Path:
    """Return the path to a cached JPEG thumbnail, generating it on first request."""
    size = int(size)
    cache_dir = _cache_dir()
    cache_file = Path(cache_dir) / f"{_cache_key(abs_path, size)}.jpg"
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file

    img = _render_thumbnail(abs_path, size)  # uint8 BGR or grayscale
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")

    tmp = cache_file.with_suffix(".jpg.tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(cache_file)
    _prune_cache(cache_dir)
    return cache_file


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_thumbnail(abs_path: str, size: int) -> np.ndarray:
    ok, kind = is_image(abs_path)
    if not ok:
        raise ValueError(f"Not a supported image: {abs_path}")
    if kind == "image":
        import cv2
        arr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError(f"Could not read image: {abs_path}")
    elif kind == "tiff":
        arr = _read_tiff_plane(abs_path)
    else:  # zarr
        arr = _read_zarr_plane(abs_path)
    return _to_display(arr, size)


def _read_tiff_plane(abs_path: str) -> np.ndarray:
    """Read a representative 2D plane cheaply (smallest pyramid level, else first IFD)."""
    import tifffile
    with tifffile.TiffFile(abs_path) as tf:
        series = tf.series[0]
        levels = getattr(series, "levels", None)
        if levels and len(levels) > 1:
            # Pyramidal OME-TIFF: the last level is the smallest -> fast.
            return _reduce_to_2d(levels[-1].asarray())
        # Non-pyramidal: read a single IFD/plane (bounded), not the whole stack.
        return _reduce_to_2d(tf.pages[0].asarray())


def _read_zarr_plane(abs_path: str) -> np.ndarray:
    """Read the smallest multiscale level of an OME-Zarr store (already tiny)."""
    import zarr
    g = zarr.open(abs_path, mode="r")
    arr = None
    try:
        datasets = dict(g.attrs)["multiscales"][0]["datasets"]
        arr = g[datasets[-1]["path"]]  # last = lowest resolution
    except Exception:
        # Fallback: pick the smallest array anywhere in the store.
        smallest = None
        try:
            for _name, node in g.arrays():
                n = int(np.prod(node.shape))
                if smallest is None or n < smallest[0]:
                    smallest = (n, node)
        except Exception:
            pass
        if smallest is not None:
            arr = smallest[1]
    if arr is None:
        raise ValueError(f"No readable array in zarr store: {abs_path}")
    return _reduce_to_2d(np.asarray(arr[...]))


def _reduce_to_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse an N-D array to a 2D (grayscale) or HxWx3 (color) plane."""
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr[..., :3]  # HWC color
    # Take the middle slice along leading axes until 2D (middle Z, first C/T).
    while arr.ndim > 2:
        arr = arr[arr.shape[0] // 2]
        arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return arr[..., :3]
    return arr


def _autocontrast(arr: np.ndarray) -> np.ndarray:
    """Percentile (1-99%) stretch to uint8 for display."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def _to_display(arr: np.ndarray, size: int) -> np.ndarray:
    import cv2
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # Color: stretch to uint8 if needed; keep channel order (cv2 writes BGR-as-is).
        img = arr if arr.dtype == np.uint8 else _autocontrast(arr)
    else:
        gray = _autocontrast(arr)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if max(h, w) > size:
        scale = size / float(max(h, w))
        img = cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def extract_metadata(abs_path: str) -> Dict[str, Any]:
    ok, kind = is_image(abs_path)
    meta: Dict[str, Any] = {
        "name": os.path.basename(abs_path.rstrip("/")),
        "isImage": ok,
        "imageType": kind,
        "sizeBytes": _path_size(abs_path),
    }
    try:
        if kind == "tiff":
            meta.update(_tiff_metadata(abs_path))
        elif kind == "zarr":
            meta.update(_zarr_metadata(abs_path))
    except Exception as exc:  # never fail the panel — surface the reason instead
        meta["metadataError"] = str(exc)
    return meta


def _path_size(p: str) -> int:
    try:
        if os.path.isdir(p):
            total = 0
            for dp, _dn, fn in os.walk(p):
                for f in fn:
                    try:
                        total += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
            return total
        return os.path.getsize(p)
    except OSError:
        return 0


def _tiff_metadata(abs_path: str) -> Dict[str, Any]:
    import tifffile
    md: Dict[str, Any] = {"format": "OME-TIFF" if abs_path.lower().endswith((".ome.tif", ".ome.tiff")) else "TIFF"}
    with tifffile.TiffFile(abs_path) as tf:
        series = tf.series[0]
        md["shape"] = [int(x) for x in series.shape]
        md["axes"] = str(series.axes)
        md["dtype"] = str(series.dtype)
        ome = tf.ome_metadata
        if ome:
            md.update(_parse_ome_xml(ome))
        else:
            desc = tf.pages[0].description or ""
            if desc:
                md["description"] = desc[:2000]
    return md


def _parse_ome_xml(xml_str: str) -> Dict[str, Any]:
    """Pull the user-facing fields out of an OME-XML string (namespace-agnostic)."""
    out: Dict[str, Any] = {}

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    root = ET.fromstring(xml_str)
    image = next((e for e in root.iter() if local(e.tag) == "Image"), None)
    if image is None:
        return out
    if image.get("AcquisitionDate"):
        out["acquisitionDate"] = image.get("AcquisitionDate")
    acq = next((e for e in image.iter() if local(e.tag) == "AcquisitionDate"), None)
    if acq is not None and acq.text:
        out["acquisitionDate"] = acq.text

    pixels = next((e for e in image.iter() if local(e.tag) == "Pixels"), None)
    if pixels is not None:
        dims = {k: pixels.get(f"Size{k}") for k in ("X", "Y", "Z", "C", "T")}
        out["dimensions"] = {k: int(v) for k, v in dims.items() if v is not None}
        ps = {}
        for axis in ("X", "Y", "Z"):
            val = pixels.get(f"PhysicalSize{axis}")
            if val is not None:
                ps[axis.lower()] = float(val)
        if ps:
            out["pixelSizeUm"] = ps
            unit = pixels.get("PhysicalSizeXUnit")
            if unit:
                out["pixelSizeUnit"] = unit
        if pixels.get("Type"):
            out["pixelType"] = pixels.get("Type")

        channels = []
        for ch in (e for e in pixels.iter() if local(e.tag) == "Channel"):
            channels.append(ch.get("Name") or ch.get("ID") or f"Channel {len(channels)}")
        if channels:
            out["channels"] = channels

        plane = next((e for e in pixels.iter() if local(e.tag) == "Plane"), None)
        if plane is not None:
            if plane.get("ExposureTime") is not None:
                out["exposure"] = {
                    "value": float(plane.get("ExposureTime")),
                    "unit": plane.get("ExposureTimeUnit") or "s",
                }
            pos = {a.lower(): plane.get(f"Position{a}") for a in ("X", "Y", "Z")}
            pos = {k: float(v) for k, v in pos.items() if v is not None}
            if pos:
                out["stagePosition"] = pos
    return out


def _zarr_metadata(abs_path: str) -> Dict[str, Any]:
    import zarr
    md: Dict[str, Any] = {"format": "OME-Zarr"}
    g = zarr.open(abs_path, mode="r")
    attrs = dict(g.attrs)
    multiscales = attrs.get("multiscales")
    if multiscales:
        ms0 = multiscales[0]
        axes = ms0.get("axes", [])
        md["axes"] = "".join(a.get("name", "") for a in axes) if axes else None
        datasets = ms0.get("datasets", [])
        if datasets:
            try:
                full = g[datasets[0]["path"]]
                md["shape"] = [int(x) for x in full.shape]
                md["dtype"] = str(full.dtype)
            except Exception:
                pass
            # pixel size from the level-0 scale transform, mapped onto spatial axes
            try:
                scale = datasets[0]["coordinateTransformations"][0]["scale"]
                axis_names = [a.get("name", "").lower() for a in axes]
                ps = {ax: float(scale[i]) for i, ax in enumerate(axis_names) if ax in ("x", "y", "z")}
                if ps:
                    md["pixelSizeUm"] = ps
            except Exception:
                pass
    omero = attrs.get("omero")
    if isinstance(omero, dict) and omero.get("channels"):
        md["channels"] = [c.get("label") or c.get("color") or f"Channel {i}"
                          for i, c in enumerate(omero["channels"])]
    return md
