"""OME-Zarr and OME-TIFF output that standard readers open.

- ``NgffImage``: one OME-NGFF image (TCZYX). NGFF 0.4 on zarr v2 by default,
  NGFF 0.5 on zarr v3 on request. Planes go in by (t, c, z) and an optional YX
  offset, so the same class serves single fields and stitched canvases. T grows
  as needed. ``finalize`` builds the resolution pyramid; ``export_ome_tiff``
  writes the same data as a tiled OME-TIFF.
- ``OmeStore``: the frame sink of a run. One image per stage field and capture
  node; an NGFF plate when the fields come from a well loop. Writes on its own
  thread so acquisition does not wait for compression.

Coordinates: µm in the stage user frame. The image translation is the
top-left pixel of the first plane; the field centre is the stage position.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numcodecs
import numpy as np
import tifffile
import zarr

NGFF_VERSIONS = ("0.4", "0.5")
_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
_COLORS = (("405", "0000FF"), ("488", "00FF00"), ("561", "FFFF00"), ("640", "FF0000"),
           ("rgb_r", "FF0000"), ("rgb_g", "00FF00"), ("rgb_b", "0000FF"))


def channel_color(name: str) -> str:
    """A display colour for a channel name (white when nothing matches)."""
    low = name.lower()
    return next((color for key, color in _COLORS if key in low), "FFFFFF")


@dataclass
class ImageSpec:
    shape: Tuple[int, int, int, int, int]                # T, C, Z, Y, X
    dtype: str = "uint16"
    pixel_size_um: float = 1.0
    z_step_um: float = 1.0
    t_step_s: float = 1.0
    origin_um: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # x, y, z
    channel_names: Sequence[str] = ()
    channel_colors: Sequence[str] = ()      # hex RRGGBB; default from the names
    name: str = "image"
    chunks_yx: Optional[Tuple[int, int]] = None   # default: up to 1024 x 1024


def _multiscales(spec: ImageSpec, levels: int, version: str) -> dict:
    ox, oy, oz = spec.origin_um
    axes = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"}] + \
           [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"]
    datasets = [{"path": str(level), "coordinateTransformations": [
        {"type": "scale", "scale": [spec.t_step_s, 1.0, spec.z_step_um,
                                    spec.pixel_size_um * 2 ** level,
                                    spec.pixel_size_um * 2 ** level]},
        {"type": "translation", "translation": [0.0, 0.0, oz, oy, ox]}]} for level in range(levels)]
    ms = {"name": spec.name, "axes": axes, "datasets": datasets}
    return {"version": version, **ms} if version == "0.4" else ms


def _omero(spec: ImageSpec, n_channels: int, version: str) -> dict:
    top = float(np.iinfo(spec.dtype).max) if np.dtype(spec.dtype).kind in "ui" else 1.0
    names = list(spec.channel_names) + [f"C{i}" for i in range(len(spec.channel_names), n_channels)]
    colors = list(spec.channel_colors) + [channel_color(n) for n in names[len(spec.channel_colors):]]
    channels = [{"label": name, "color": color, "active": True, "coefficient": 1.0,
                 "family": "linear", "inverted": False,
                 "window": {"min": 0.0, "max": top, "start": 0.0, "end": top}}
                for name, color in zip(names, colors)]
    omero = {"channels": channels, "rdefs": {"model": "color", "defaultT": 0,
                                             "defaultZ": spec.shape[2] // 2}}
    return {"version": version, **omero} if version == "0.4" else omero


class NgffImage:
    """One OME-NGFF image in a zarr group."""

    BAND_ROWS = 4096   # rows read at a time when building the pyramid (bounded RAM)

    def __init__(self, group: zarr.Group, spec: ImageSpec, version: str = "0.4"):
        if version not in NGFF_VERSIONS:
            raise ValueError(f"NGFF version must be one of {NGFF_VERSIONS}")
        self.group, self.spec, self.version = group, spec, version
        self.levels = 1
        self.t_written = 0          # time points that received data
        _, _, _, y, x = spec.shape
        self.chunks_yx = spec.chunks_yx or (min(y, 1024), min(x, 1024))
        self.array = self._create("0", spec.shape, (1, 1, 1) + tuple(self.chunks_yx))
        self._write_attrs()

    @classmethod
    def create(cls, path, spec: ImageSpec, version: str = "0.4") -> "NgffImage":
        group = zarr.open_group(str(path), mode="w", zarr_format=2 if version == "0.4" else 3)
        return cls(group, spec, version)

    def _create(self, name, shape, chunks):
        kwargs = {"shape": shape, "chunks": chunks, "dtype": self.spec.dtype, "fill_value": 0,
                  "overwrite": True}
        if self.version == "0.4":
            kwargs.update(compressors=_COMPRESSOR,
                          chunk_key_encoding={"name": "v2", "separator": "/"})
        else:
            kwargs.update(dimension_names=list("tczyx"))
        return self.group.create_array(name, **kwargs)

    def _write_attrs(self):
        ms = _multiscales(self.spec, self.levels, self.version)
        omero = _omero(self.spec, self.array.shape[1], self.version)
        if self.version == "0.4":
            self.group.attrs.update({"multiscales": [ms], "omero": omero})
        else:
            ome = {"version": "0.5", "multiscales": [ms], "omero": omero}
            self.group.attrs.update({"ome": ome})

    def write(self, plane: np.ndarray, t: int, c: int, z: int, y0: int = 0, x0: int = 0) -> None:
        if t >= self.array.shape[0]:
            self.array.resize((t + 1,) + tuple(self.array.shape[1:]))
        h, w = plane.shape[:2]
        self.array[t, c, z, y0:y0 + h, x0:x0 + w] = plane
        self.t_written = max(self.t_written, t + 1)

    def finalize(self, min_size: int = 256, max_levels: int = 5) -> None:
        """Trim T to what was written, add 2x levels (stride decimation, row bands)."""
        source = self.array
        if 0 < self.t_written < source.shape[0]:   # a run that ended early
            source.resize((self.t_written,) + tuple(source.shape[1:]))
        _, _, _, y, x = source.shape
        while self.levels < max_levels and min(y, x) // 2 ** self.levels >= min_size:
            step = 2 ** self.levels
            shape = source.shape[:3] + (-(-y // step), -(-x // step))
            chunks = (1, 1, 1) + tuple(min(s, c) for s, c in zip(shape[3:], self.chunks_yx))
            level = self._create(str(self.levels), shape, chunks)
            band = max(step, (self.BAND_ROWS // step) * step)
            for t, c, z in np.ndindex(*shape[:3]):
                for y0 in range(0, y, band):
                    rows = source[t, c, z, y0:y0 + band, :][::step, ::step]
                    level[t, c, z, y0 // step:y0 // step + rows.shape[0], :] = rows
            self.levels += 1
        self.spec.shape = tuple(source.shape)
        self._write_attrs()

    def export_ome_tiff(self, path) -> Path:
        """Write level 0 as a tiled, compressed OME-TIFF (planes streamed, not held in RAM)."""
        t, c, z, y, x = self.array.shape
        spec = self.spec
        names = list(spec.channel_names) + [f"C{i}" for i in range(len(spec.channel_names), c)]
        ox, oy, oz = spec.origin_um
        planes = [(ti, ci, zi) for ti in range(t) for ci in range(c) for zi in range(z)]
        metadata = {
            "axes": "TCZYX", "Name": spec.name,
            "PhysicalSizeX": spec.pixel_size_um, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": spec.pixel_size_um, "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": spec.z_step_um, "PhysicalSizeZUnit": "µm",
            "TimeIncrement": spec.t_step_s, "TimeIncrementUnit": "s",
            "Channel": {"Name": names},
            "Plane": {"PositionX": [ox] * len(planes), "PositionY": [oy] * len(planes),
                      "PositionZ": [oz + zi * spec.z_step_um for _, _, zi in planes],
                      "PositionXUnit": ["µm"] * len(planes), "PositionYUnit": ["µm"] * len(planes),
                      "PositionZUnit": ["µm"] * len(planes),
                      "DeltaT": [ti * spec.t_step_s for ti, _, _ in planes]},
        }
        path = Path(path)
        tile = (min(512, -(-y // 16) * 16), min(512, -(-x // 16) * 16))
        with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tif:
            tif.write((self.array[p] for p in planes), shape=(t, c, z, y, x), dtype=spec.dtype,
                      metadata=metadata, photometric="minisblack", tile=tile,
                      compression="zlib", compressionargs={"level": 1}, predictor=True)
        return path


# --------------------------------------------------------------------------- #
# Plates
# --------------------------------------------------------------------------- #
def _row_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


class NgffPlate:
    """An NGFF plate: wells appear as their first field is written."""

    def __init__(self, path, name: str, version: str = "0.4"):
        self.version = version
        self.group = zarr.open_group(str(path), mode="w", zarr_format=2 if version == "0.4" else 3)
        self.name = name
        self.wells: Dict[Tuple[int, int], List[int]] = {}

    def field_group(self, row: int, col: int, field_index: int) -> zarr.Group:
        fields = self.wells.setdefault((row, col), [])
        group = self.group.require_group(f"{_row_name(row)}/{col + 1}/{field_index}")
        if field_index not in fields:
            fields.append(field_index)
            self._write_attrs(row, col)
        return group

    def _wrap(self, key: str, value: dict) -> dict:
        if self.version == "0.4":
            return {key: {"version": "0.4", **value}}
        return {"ome": {"version": "0.5", key: value}}

    def _write_attrs(self, row: int, col: int) -> None:
        n_rows = max(r for r, _ in self.wells) + 1
        n_cols = max(c for _, c in self.wells) + 1
        plate = {"name": self.name,
                 "rows": [{"name": _row_name(r)} for r in range(n_rows)],
                 "columns": [{"name": str(c + 1)} for c in range(n_cols)],
                 "wells": [{"path": f"{_row_name(r)}/{c + 1}", "rowIndex": r, "columnIndex": c}
                           for r, c in sorted(self.wells)],
                 "field_count": max(len(f) for f in self.wells.values()),
                 "acquisitions": [{"id": 0}]}
        self.group.attrs.update(self._wrap("plate", plate))
        well = {"images": [{"path": str(f), "acquisition": 0}
                           for f in sorted(self.wells[(row, col)])]}
        self.group[f"{_row_name(row)}/{col + 1}"].attrs.update(self._wrap("well", well))


# --------------------------------------------------------------------------- #
# The frame sink of a run
# --------------------------------------------------------------------------- #
@dataclass
class FrameInfo:
    """Where a frame belongs. Built by the engine from the active loops."""

    node_id: str                                  # the capture node
    index: Dict[str, int]                         # useq axes: t, p, g, z, c
    sizes: Dict[str, Optional[int]]               # known loop counts per axis
    channel_names: List[str]
    pixel_size_um: float
    position: Optional[Dict[str, Any]] = None     # name, x, y, z (+ row, col for wells)
    tile: Optional[Dict[str, Any]] = None         # name, x, y (absolute) for grid tiles
    z_um: Optional[float] = None
    z_step_um: Optional[float] = None
    t_step_s: Optional[float] = None
    stage_xy: Optional[Tuple[float, float]] = None   # when there is no position loop


@dataclass
class _Field:
    image: NgffImage
    tiff_name: str


class OmeStore:
    """Routes frames to one NgffImage per (capture node, position, tile)."""

    def __init__(self, root, formats: Sequence[str] = ("ome-zarr",), layout: str = "auto",
                 version: str = "0.4", queue_size: int = 32):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.formats, self.layout, self.version = tuple(formats), layout, version
        self.fields: Dict[Tuple, _Field] = {}
        self.plates: Dict[str, NgffPlate] = {}
        self.frames_written = 0
        self.error: Optional[BaseException] = None
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._drain, name="aufgaben-writer", daemon=True)
        self._thread.start()

    # -- public ---------------------------------------------------------------
    def write(self, frame: np.ndarray, info: FrameInfo) -> None:
        if self.error:
            raise RuntimeError(f"writer failed: {self.error}") from self.error
        self._queue.put((np.asarray(frame).copy(), info))

    def close(self) -> List[Path]:
        """Flush, build pyramids, export OME-TIFFs. Returns the files written."""
        self._queue.put(None)
        self._thread.join()
        written = []
        for f in self.fields.values():
            f.image.finalize()
            if "ome-tiff" in self.formats:
                written.append(f.image.export_ome_tiff(self.root / f.tiff_name))
        if self.error:
            raise RuntimeError(f"writer failed: {self.error}") from self.error
        return written

    # -- internals ------------------------------------------------------------
    def _drain(self) -> None:
        while (item := self._queue.get()) is not None:
            if self.error:
                continue
            try:
                self._write(*item)
            except BaseException as e:  # reported by write()/close()
                self.error = e

    def _write(self, frame: np.ndarray, info: FrameInfo) -> None:
        planes = [frame] if frame.ndim == 2 else [frame[..., k] for k in range(frame.shape[-1])]
        f = self._field(frame, info, len(planes))
        t, z = info.index.get("t", 0), info.index.get("z", 0)
        for k, plane in enumerate(planes):
            f.image.write(plane, t, info.index.get("c", 0) * len(planes) + k, z)
        self.frames_written += 1

    def _field(self, frame: np.ndarray, info: FrameInfo, samples: int) -> _Field:
        pos, tile = info.position or {}, info.tile or {}
        key = (info.node_id, pos.get("name"), tile.get("name"))
        if key in self.fields:
            return self.fields[key]
        h, w = frame.shape[:2]
        cx, cy = (tile.get("x"), tile.get("y")) if tile else (pos.get("x"), pos.get("y"))
        if cx is None:
            cx, cy = info.stage_xy or (0.0, 0.0)
        px = info.pixel_size_um
        z_index, z_step = info.index.get("z", 0), info.z_step_um or 1.0
        z0 = (info.z_um if info.z_um is not None else (pos.get("z") or 0.0)) - z_index * z_step
        names = [f"{n}_rgb_{s}" for n in info.channel_names for s in "rgb"] if samples == 3 \
            else list(info.channel_names)
        n_c = (info.sizes.get("c") or 1) * samples
        spec = ImageSpec(
            shape=(info.sizes.get("t") or 1, n_c, info.sizes.get("z") or 1, h, w),
            dtype=str(frame.dtype), pixel_size_um=px, z_step_um=z_step,
            t_step_s=info.t_step_s or 1.0,
            origin_um=(cx - w * px / 2, cy - h * px / 2, z0), channel_names=names,
            name="_".join(str(p) for p in key[1:] if p) or key[0])
        plate = "row" in pos and self.layout in ("auto", "plate")
        if plate:
            if info.node_id not in self.plates:
                self.plates[info.node_id] = NgffPlate(
                    self.root / f"{info.node_id}.ome.zarr", info.node_id, self.version)
            group = self.plates[info.node_id].field_group(
                pos["row"], pos["col"], info.index.get("g", 0))
            image = NgffImage(group, spec, self.version)
            tiff = f"{info.node_id}_{pos['name']}_{info.index.get('g', 0)}.ome.tif"
        else:
            stem = "_".join(str(p) for p in (pos.get("name"), tile.get("name"), info.node_id) if p)
            image = NgffImage.create(self.root / f"{stem}.ome.zarr", spec, self.version)
            tiff = f"{stem}.ome.tif"
        self.fields[key] = _Field(image, tiff)
        return self.fields[key]
