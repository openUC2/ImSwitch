"""OME-Zarr and OME-TIFF output that standard readers open.

- ``NgffImage``: one OME-NGFF image (TCZYX). NGFF 0.4 on zarr v2 by default,
  NGFF 0.5 on zarr v3 on request. Planes go in by (t, c, z) and an optional YX
  offset, so the same class serves single fields and stitched canvases. T grows
  as needed. ``finalize`` builds the resolution pyramid; ``export_ome_tiff``
  writes the same data, pyramid included, as a tiled OME-TIFF.
- ``save_image``: one numpy image to an OME-TIFF or OME-Zarr file. With
  ``NgffImage`` (and its ``export_ome_tiff``) this is how ImSwitch writes image
  files; nothing else calls tifffile or zarr to write images.
- ``ome_xml``: the OME-XML of an image (ome-types), shared by every OME-TIFF.
- ``OmeStore``: the frame sink of a run. One image per stage field and capture
  node; an NGFF plate when the fields come from a well loop. Writes on its own
  thread so acquisition does not wait for compression.

Every group's attributes are checked against the ome-zarr-models schema of
their NGFF version before they are written, so invalid metadata fails loudly.

Coordinates: µm in the stage user frame. The NGFF translation and the OME
plane positions are the top-left pixel corner; the field centre is the stage
position. An unknown pixel size or position is left out of the metadata rather
than written as a guess.

Readers: OME-TIFF opens as OME in Fiji/Bio-Formats (the XML is UTF-8, which
Bio-Formats needs for the µ in units), napari (tifffile) and the openUC2
napari plugin; OME-Zarr (NGFF 0.4, zarr v2) in napari-ome-zarr, Fiji's
N5/OME-NGFF importer and the plugin.

Each image also carries ``attrs["imswitch"]``: ``node``, ``position`` and
``tile`` for raw fields, ``derived`` ("stitched", or a task's image name) for
computed images, so readers can rebuild the scan without parsing file names.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numcodecs
import numpy as np
import tifffile
import zarr
from ome_types import OME
from ome_types.model import (AnnotationRef, Channel, Image, Map, MapAnnotation, Pixels, Plane,
                             StructuredAnnotations, TiffData)
from ome_zarr_models.v04.hcs import HCSAttrs as _Hcs04
from ome_zarr_models.v04.image import ImageAttrs as _Image04
from ome_zarr_models.v04.well import WellAttrs as _Well04
from ome_zarr_models.v05.hcs import HCSAttrs as _Hcs05
from ome_zarr_models.v05.image import ImageAttrs as _Image05
from ome_zarr_models.v05.well import WellAttrs as _Well05

NGFF_VERSIONS = ("0.4", "0.5")
_SCHEMAS = {"0.4": {"image": _Image04, "plate": _Hcs04, "well": _Well04},
            "0.5": {"image": _Image05, "plate": _Hcs05, "well": _Well05}}
# Zstd, not Blosc: as fast at level 1 (~600 MB/s), and Fiji's N5/OME-NGFF reader decodes it
# on every platform (its Blosc needs a native library that is missing on macOS arm64).
_COMPRESSOR = numcodecs.Zstd(level=1)
_TIFF_OPTIONS = dict(photometric="minisblack", compression="zlib", compressionargs={"level": 1},
                     predictor=True)
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
    pixel_size_um: Optional[float] = 1.0       # None: not calibrated, left out
    z_step_um: float = 1.0
    t_step_s: float = 1.0
    origin_um: Optional[Tuple[float, float, Optional[float]]] = (0.0, 0.0, 0.0)  # top-left x, y, z
    channel_names: Sequence[str] = ()
    channel_colors: Sequence[str] = ()      # hex RRGGBB; default from the names
    name: str = "image"
    chunks_yx: Optional[Tuple[int, int]] = None   # default: up to 1024 x 1024
    exposure_ms: List[Optional[float]] = field(default_factory=list)   # per channel
    bit_depth: Optional[int] = None         # camera bits; sets the display window
    annotations: Dict[str, Any] = field(default_factory=dict)   # acquisition key/values


    def channels(self, n: int) -> List[Tuple[str, str]]:
        """(name, colour) for n channels, filling in what the spec leaves out."""
        names = list(self.channel_names)[:n] + [f"C{i}" for i in range(len(self.channel_names), n)]
        colors = list(self.channel_colors)[:n]
        return list(zip(names, colors + [channel_color(nm) for nm in names[len(colors):]]))

    def display_max(self) -> float:
        if self.bit_depth:
            return float(2 ** self.bit_depth - 1)
        return float(np.iinfo(self.dtype).max) if np.dtype(self.dtype).kind in "ui" else 1.0


def _flat(values: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Nested metadata as flat "a/b" -> text pairs (OME MapAnnotation, zarr attrs)."""
    out: Dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, f"{name}/"))
        elif value is not None:
            out[name] = str(value)
    return out


def _group_attrs(version: str, kind: str, body: dict) -> dict:
    """Attributes of an NGFF group, checked against the ome-zarr-models schema."""
    if version == "0.5":
        body = {"version": "0.5", **body}
    _SCHEMAS[version][kind].model_validate(body)
    return body if version == "0.4" else {"ome": body}


def _versioned(version: str, body: dict) -> dict:
    """NGFF 0.4 carries the version inside each metadata block; 0.5 once at the top."""
    return {"version": version, **body} if version == "0.4" else body


def _multiscales(spec: ImageSpec, levels: int, version: str) -> dict:
    ox, oy, oz = spec.origin_um or (0.0, 0.0, 0.0)
    oz = oz or 0.0
    px = spec.pixel_size_um or 1.0
    unit = {"unit": "micrometer"} if spec.pixel_size_um else {}
    axes = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"}] + \
           [{"name": a, "type": "space", **unit} for a in "zyx"]
    datasets = [{"path": str(level), "coordinateTransformations": [
        {"type": "scale", "scale": [spec.t_step_s, 1.0, spec.z_step_um,
                                    px * 2 ** level, px * 2 ** level]},
        {"type": "translation", "translation": [0.0, 0.0, oz, oy, ox]}]} for level in range(levels)]
    return _versioned(version, {"name": spec.name, "axes": axes, "datasets": datasets})


def _omero(spec: ImageSpec, n_channels: int, version: str) -> dict:
    top = spec.display_max()
    channels = [{"label": name, "color": color, "active": True, "coefficient": 1.0,
                 "family": "linear", "inverted": False,
                 "window": {"min": 0.0, "max": top, "start": 0.0, "end": top}}
                for name, color in spec.channels(n_channels)]
    return _versioned(version, {"channels": channels, "rdefs": {
        "model": "color", "defaultT": 0, "defaultZ": spec.shape[2] // 2}})


def ome_xml(spec: ImageSpec, rgb: bool = False) -> str:
    """OME-XML for an image whose planes are stored in T, C, Z order (XYZCT).

    ``rgb``: every three channels are one interleaved RGB channel (SamplesPerPixel 3),
    as in a TIFF written with photometric RGB. Positions are the top-left pixel
    corner in µm; exposure per channel when the spec has it. Store it UTF-8
    encoded (``_write_ome_tiff`` does): Bio-Formats rejects the units otherwise.
    """
    t, c, z, y, x = spec.shape
    samples = 3 if rgb else 1
    channels = spec.channels(c)[::samples]
    exposures = list(spec.exposure_ms) + [None] * len(channels)

    def where(zi: int) -> dict:
        if spec.origin_um is None:
            return {}
        ox, oy, oz = spec.origin_um
        z = {} if oz is None else {"position_z": oz + zi * spec.z_step_um, "position_z_unit": "µm"}
        return {"position_x": ox, "position_x_unit": "µm", "position_y": oy,
                "position_y_unit": "µm", **z}

    planes = [Plane(the_t=ti, the_c=ci, the_z=zi, delta_t=ti * spec.t_step_s, delta_t_unit="s",
                    exposure_time=exposures[ci], exposure_time_unit="ms", **where(zi))
              for ti in range(t) for ci in range(len(channels)) for zi in range(z)]
    px = spec.pixel_size_um
    sizes = {} if px is None else {
        "physical_size_x": px, "physical_size_x_unit": "µm", "physical_size_y": px,
        "physical_size_y_unit": "µm", "physical_size_z": spec.z_step_um,
        "physical_size_z_unit": "µm"}
    pixels = Pixels(
        dimension_order="XYZCT", type=str(np.dtype(spec.dtype)), significant_bits=spec.bit_depth,
        size_x=x, size_y=y, size_z=z, size_c=c, size_t=t, **sizes,
        time_increment=spec.t_step_s, time_increment_unit="s",
        channels=[Channel(name=name.removesuffix("_rgb_r") if rgb else name,
                          samples_per_pixel=samples, **({} if rgb else {"color": f"#{color}"}))
                  for name, color in channels],
        planes=planes, tiff_data_blocks=[TiffData(plane_count=len(planes))])
    if not spec.annotations:
        return OME(images=[Image(name=spec.name, pixels=pixels)], creator="ImSwitch").to_xml()
    note = MapAnnotation(namespace="openuc2.org/imswitch/acquisition", value=Map(ms=[
        Map.M(k=k, value=v) for k, v in _flat(spec.annotations).items()]))
    image = Image(name=spec.name, pixels=pixels, annotation_refs=[AnnotationRef(id=note.id)])
    return OME(images=[image], structured_annotations=StructuredAnnotations(map_annotations=[note]),
               creator="ImSwitch").to_xml()


def _tiff_tile(y: int, x: int) -> Tuple[int, int]:
    return min(512, -(-y // 16) * 16), min(512, -(-x // 16) * 16)


def _tiff_tiles(level, planes, tile: Tuple[int, int]):
    """TIFF tiles of each plane in write order, edge tiles zero-padded to full size.

    tifffile takes an iterator of *tiles* when writing tiled pages; one plane is
    read at a time, so memory stays at one plane.
    """
    th, tw = tile
    for p in planes:
        plane = np.asarray(level[p])
        for y0 in range(0, plane.shape[0], th):
            for x0 in range(0, plane.shape[1], tw):
                block = plane[y0:y0 + th, x0:x0 + tw]
                if block.shape[:2] != (th, tw):
                    pad = [(0, th - block.shape[0]), (0, tw - block.shape[1])]
                    block = np.pad(block, pad + [(0, 0)] * (block.ndim - 2))
                yield block


def _write_ome_tiff(path, levels, spec: ImageSpec, rgb: bool = False) -> Path:
    """Tiled, compressed OME-TIFF; levels after the first become SubIFDs (a pyramid).

    ``levels``: arrays indexed (t, c, z) -> plane (Y, X), or (Y, X, 3) with ``rgb``.
    Planes are streamed tile by tile, so zarr-backed levels never load whole.
    """
    planes = list(np.ndindex(*levels[0].shape[:3]))
    samples = (3,) if rgb else ()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, bigtiff=True, ome=False) as tif:
        for i, level in enumerate(levels):
            y, x = level.shape[3:5]
            tile = _tiff_tile(y, x)
            first = {"subifds": len(levels) - 1,
                     "description": ome_xml(spec, rgb=rgb).encode("utf-8")} if i == 0 \
                else {"subfiletype": 1}
            tif.write(_tiff_tiles(level, planes, tile), shape=(len(planes), y, x) + samples,
                      dtype=spec.dtype, tile=tile, metadata=None, **first,
                      **{**_TIFF_OPTIONS, "photometric": "rgb" if rgb else "minisblack"})
    return path


def _pyramid(array: np.ndarray, min_size: int = 256, max_levels: int = 5) -> list:
    """``array`` (T, C, Z, Y, X[, S]) and its 2x stride-decimated levels."""
    y, x = array.shape[3:5]
    levels = [array]
    while len(levels) < max_levels and min(y, x) // 2 ** len(levels) >= min_size:
        step = 2 ** len(levels)
        levels.append(array[:, :, :, ::step, ::step])
    return levels


def save_image(path, data: np.ndarray, axes: Optional[str] = None, *,
               pixel_size_um: Optional[float] = None, z_step_um: float = 1.0,
               t_step_s: float = 1.0,
               origin_um: Optional[Tuple[float, float, Optional[float]]] = None,
               channel_names: Sequence[str] = (), channel_colors: Sequence[str] = (),
               exposure_ms: Sequence[Optional[float]] = (), bit_depth: Optional[int] = None,
               name: Optional[str] = None, annotations: Optional[Dict[str, Any]] = None,
               version: str = "0.4") -> Path:
    """Write one image as OME-TIFF (``*.tif``, ``*.tiff``) or OME-Zarr (``*.zarr``).

    Args:
        path: The file or store; its suffix picks the format.
        data: The pixels. Its axes are named by ``axes``, letters from "TCZYX" plus
            "S" for RGB samples: "YX", "ZYX", "CYX", "TZYX", "TCZYX", "YXS" ...
            Default "YX" for 2-D data and "YXS" for (Y, X, 3).
        pixel_size_um: Pixel size; None when the camera is not calibrated.
        z_step_um: Z spacing.
        t_step_s: Time between time points.
        origin_um: Stage position (x, y, z) of the top-left pixel; None when unknown
            (z alone may be None).
        channel_names: One per channel (per C index).
        channel_colors: Hex RRGGBB per channel; default from the names.
        exposure_ms: Per channel.
        bit_depth: Camera bits; sets the display window and SignificantBits.
        name: Image name; default the file name.
        annotations: Acquisition key/values (nested dicts are flattened); an OME
            MapAnnotation in TIFF, ``attrs["acquisition"]`` in zarr.
        version: NGFF version for OME-Zarr.

    Returns the path written. Large images get a pyramid (SubIFDs or zarr levels).
    """
    path = Path(path)
    array = np.asarray(data)
    axes = (axes or ("YXS" if array.ndim == 3 and array.shape[-1] == 3 else "YX")).upper()
    if len(axes) != array.ndim or len(set(axes)) != len(axes) or set(axes) - set("TCZYXS") \
            or not {"Y", "X"} <= set(axes):
        raise ValueError(f"axes {axes!r} do not describe data of shape {array.shape}")
    rgb = "S" in axes
    present = [a for a in "TCZYXS" if a in axes]
    array = array.transpose([axes.index(a) for a in present])
    full = "TCZYX" + ("S" if rgb else "")
    array = array.reshape([array.shape[present.index(a)] if a in present else 1 for a in full])
    t, c, z, y, x = array.shape[:5]
    names = list(channel_names) or [f"C{i}" for i in range(c)]
    spec = ImageSpec(shape=(t, c * (3 if rgb else 1), z, y, x), dtype=str(array.dtype),
                     pixel_size_um=pixel_size_um, z_step_um=z_step_um, t_step_s=t_step_s,
                     origin_um=origin_um, bit_depth=bit_depth,
                     channel_names=[f"{n}_rgb_{s}" for n in names for s in "rgb"] if rgb
                     else names, channel_colors=() if rgb else channel_colors,
                     exposure_ms=[e for e in exposure_ms for _ in range(3 if rgb else 1)],
                     name=name or path.name.split(".")[0], annotations=dict(annotations or {}))
    if path.suffix.lower() in (".tif", ".tiff"):
        return _write_ome_tiff(path, _pyramid(array), spec, rgb=rgb)
    if path.suffix.lower() != ".zarr":
        raise ValueError(f"{path.name}: use .tif/.tiff (OME-TIFF) or .zarr (OME-Zarr)")
    image = NgffImage.create(path, spec, version)
    for ti, ci, zi in np.ndindex(t, c, z):
        planes = [array[ti, ci, zi]] if not rgb else [array[ti, ci, zi, ..., k] for k in range(3)]
        for k, plane in enumerate(planes):
            image.write(plane, ti, ci * len(planes) + k, zi)
    image.finalize()
    return path


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
        body = {"multiscales": [_multiscales(self.spec, self.levels, self.version)],
                "omero": _omero(self.spec, self.array.shape[1], self.version)}
        self.group.attrs.update(_group_attrs(self.version, "image", body))
        if self.spec.annotations:
            self.group.attrs["acquisition"] = _flat(self.spec.annotations)

    def write(self, plane: np.ndarray, t: int, c: int, z: int, y0: int = 0, x0: int = 0) -> None:
        if t >= self.array.shape[0]:   # grow in doubling steps; finalize trims the rest
            self.array.resize((max(t + 1, 2 * self.array.shape[0]),) + tuple(self.array.shape[1:]))
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
        """Write the image and its pyramid (as SubIFDs) to a tiled OME-TIFF.

        Planes are streamed from zarr, never held in memory as a whole.
        """
        return _write_ome_tiff(path, [self.group[str(i)] for i in range(self.levels)], self.spec)

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
        v = self.version
        self.group.attrs.update(_group_attrs(v, "plate", {"plate": _versioned(v, plate)}))
        well = {"images": [{"path": str(f), "acquisition": 0}
                           for f in sorted(self.wells[(row, col)])]}
        self.group[f"{_row_name(row)}/{col + 1}"].attrs.update(
            _group_attrs(v, "well", {"well": _versioned(v, well)}))


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
    exposure_ms: Optional[float] = None
    bit_depth: Optional[int] = None


@dataclass
class _Field:
    image: NgffImage
    tiff_name: str


class OmeStore:
    """Routes frames to one NgffImage per (capture node, position, tile)."""

    def __init__(self, root, formats: Sequence[str] = ("ome-zarr",), layout: str = "auto",
                 version: str = "0.4", queue_size: int = 32, stitch: Optional[str] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.formats, self.layout, self.version = tuple(formats), layout, version
        self.stitch = stitch          # None, "stage" or "ashlar": stitched image per position
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
        """Flush, build pyramids, stitch, export OME-TIFFs. Returns the TIFFs written."""
        self._queue.put(None)
        self._thread.join()
        for f in self.fields.values():
            f.image.finalize()
        images = [(f.image, f.tiff_name) for f in self.fields.values()]
        if self.stitch:
            images += self._stitched()
        written = [image.export_ome_tiff(self.root / tiff) for image, tiff in images
                   if "ome-tiff" in self.formats]
        if self.error:
            raise RuntimeError(f"writer failed: {self.error}") from self.error
        return written

    def _stitched(self) -> List[Tuple["NgffImage", str]]:
        """One stitched image per (capture node, position) that has several tiles."""
        from .stitching import stitch
        groups: Dict[Tuple, List[NgffImage]] = {}
        for (node, pos, tile), f in self.fields.items():
            if tile is not None:
                groups.setdefault((node, pos), []).append(f.image)
        out = []
        for (node, pos), tiles in groups.items():
            if len(tiles) > 1:
                stem = "_".join(str(p) for p in (pos, node, "stitched") if p)
                image = stitch(tiles, self.root / f"{stem}.ome.zarr", stem, self.stitch)
                image.group.attrs["imswitch"] = {"node": node, "position": pos,
                                                 "derived": "stitched"}
                out.append((image, f"{stem}.ome.tif"))
        return out

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
            c = info.index.get("c", 0) * len(planes) + k
            f.image.write(plane, t, c, z)
            exposures = f.image.spec.exposure_ms
            exposures.extend([None] * (c + 1 - len(exposures)))
            exposures[c] = info.exposure_ms
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
            name="_".join(str(p) for p in key[1:] if p) or key[0], bit_depth=info.bit_depth)
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
        image.group.attrs["imswitch"] = {"node": info.node_id, "position": pos.get("name"),
                                         "tile": tile.get("name")}
        self.fields[key] = _Field(image, tiff)
        return self.fields[key]
