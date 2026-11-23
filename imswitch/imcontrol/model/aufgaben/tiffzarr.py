"""Read an OME-TIFF as a read-only Zarr v2 store (NGFF 0.4), one TIFF tile per chunk.

The dataset viewer reads OME-Zarr over HTTP. With this it also opens the OME-TIFF files
ImSwitch writes (one plane per page, tiled, SubIFD pyramid) without converting them or
loading them whole: ``read(path, key)`` answers the store keys ``.zgroup``, ``.zattrs``,
``<level>/.zarray`` and ``<level>/<chunk>``. tifffile's own ``aszarr`` store is not used; it
only imports next to matching zarr releases. No ImSwitch imports.
"""
import json
import os
from functools import lru_cache

import numpy as np
import tifffile


def _pixels(tif) -> dict:
    """The OME ``Pixels`` element of the first image; empty when the file has none."""
    try:
        image = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]
        return (image[0] if isinstance(image, list) else image)["Pixels"]
    except Exception:
        return {}


def _channels(pixels: dict, rgb: bool) -> list:
    if rgb:
        return [{"label": name, "color": color} for name, color in
                (("R", "FF0000"), ("G", "00FF00"), ("B", "0000FF"))]
    found = pixels.get("Channel") or []
    out = []
    for i, ch in enumerate(found if isinstance(found, list) else [found]):
        entry = {"label": ch.get("Name") or f"C{i}"}
        if isinstance(ch.get("Color"), int):                 # OME: signed RGBA
            entry["color"] = f"{(ch['Color'] >> 8) & 0xFFFFFF:06X}"
        out.append(entry)
    return out


# ponytail: first series only (ImSwitch writes one image per file); add a series index to
# the key when multi-series files matter.
@lru_cache(maxsize=16)
def _index(path: str, mtime: float) -> dict:
    """What every key needs, read once per file version; the file is closed again."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        lead = axes.split("Y")[0]
        rgb = axes.endswith("S")
        if set(lead) - set("TCZ") or axes[len(lead):] not in ("YX", "YXS") or (rgb and "C" in lead):
            raise ValueError(f"TIFF axes {axes} are not supported (T, C, Z in any order, then YX)")
        pixels = _pixels(tif)
        levels = []
        for level in series.levels:
            key = level.keyframe
            counts = level.shape[:len(lead)]
            if (rgb and key.planarconfig != 1) or len(level.pages) != int(np.prod(counts)):
                raise ValueError("TIFF layout is not supported (one plane per page expected)")
            y, x = level.shape[len(lead):len(lead) + 2]
            samples = (key.samplesperpixel,) if rgb else ()
            tile = tuple(key.chunks[:2])
            levels.append({
                "lead": counts, "decode": key.decode, "columns": -(-x // tile[1]),
                "dtype": np.dtype(key.dtype).newbyteorder("<"),
                "shape": [*counts, *samples, y, x],
                "chunks": [*(1,) * len(lead), *samples, *tile],
                "segments": [None if p is None else (p.dataoffsets, p.databytecounts)
                             for p in level.pages]})
    names = list(lead.lower()) + (["c"] if rgb else []) + ["y", "x"]
    px = pixels.get("PhysicalSizeX")
    unit = {"unit": "micrometer"} if px else {}
    step = {"t": pixels.get("TimeIncrement") or 1.0, "c": 1.0,
            "z": pixels.get("PhysicalSizeZ") or 1.0}
    y0 = levels[0]["shape"][-2]
    datasets = [{"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [
        step[a] if a in step else (px or 1.0) * y0 / lv["shape"][-2] for a in names]}]}
        for i, lv in enumerate(levels)]
    kinds = {"t": {"type": "time", "unit": "second"}, "c": {"type": "channel"}}
    attrs = {"multiscales": [{
        "version": "0.4", "name": os.path.basename(path), "datasets": datasets,
        "axes": [{"name": a, **kinds.get(a, {"type": "space", **unit})} for a in names]}],
        "omero": {"channels": _channels(pixels, rgb)}}
    return {"attrs": attrs, "levels": levels}


def _chunk(path: str, level: dict, index: list) -> bytes:
    """One zarr chunk = one TIFF tile (or strip), decoded and padded to the full chunk."""
    n = len(level["lead"])
    th, tw = level["chunks"][-2:]
    if len(index) != len(level["shape"]) or any(i < 0 for i in index) or any(index[n:-2]) \
            or index[-2] * th >= level["shape"][-2] or index[-1] >= level["columns"]:
        raise KeyError(index)
    page = level["segments"][int(np.ravel_multi_index(index[:n], level["lead"])) if n else 0]
    out = np.zeros(level["chunks"], level["dtype"])
    segment = index[-2] * level["columns"] + index[-1]
    if page is not None and page[1][segment]:
        with open(path, "rb") as f:                # own handle: requests run in parallel
            f.seek(page[0][segment])
            data = f.read(page[1][segment])
        tile = np.moveaxis(level["decode"](data, segment)[0][0], -1, 0)    # (samples, y, x)
        out.reshape(-1, th, tw)[:, :tile.shape[1], :tile.shape[2]] = tile[:, :th, :tw]
    return out.tobytes()


def read(path, key: str) -> bytes:
    """Content of one store key of the TIFF at ``path``.

    Raises KeyError when the store has no such key and ValueError when the TIFF is laid out
    in a way this reader does not handle (axes other than T, C, Z, Y, X, planar RGB, several
    planes in one page).
    """
    path = str(path)
    index = _index(path, os.path.getmtime(path))
    if key == ".zgroup":
        return b'{"zarr_format": 2}'
    if key == ".zattrs":
        return json.dumps(index["attrs"]).encode()
    name, _, item = key.partition("/")
    try:
        level = index["levels"][int(name)] if name.isdigit() else None
        if level is None:
            raise KeyError(key)
        if item == ".zarray":
            return json.dumps({
                "zarr_format": 2, "shape": level["shape"], "chunks": level["chunks"],
                "dtype": level["dtype"].str, "compressor": None, "filters": None,
                "fill_value": 0, "order": "C", "dimension_separator": "."}).encode()
        return _chunk(path, level, [int(i) for i in item.split(".")])
    except (ValueError, IndexError) as e:          # not a chunk key, or outside the array
        raise KeyError(key) from e
