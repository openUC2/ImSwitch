"""OME-Zarr canvases into mikro, with the pyramid they already have.

ImSwitch writes a scan region as one NGFF image (``NgffImage``: TCZYX, level
0 plus 2x stride-decimated levels, built when the scan ends). mikro takes a
dataset's levels when the dataset is created and has no way to add them
later, so the levels on disk go up as they are and nothing is downsampled
again. mikro re-encodes them (zarr v3, its own chunks and shards); dask
streams them chunk by chunk, so a canvas larger than RAM is fine.

Placement: each canvas is registered in one persistent stage space (µm,
get-or-create by ``STAGE_SPACE``), at its top-left corner and with its pixel
size, so every scan of this microscope lands in the same space.

``UploadQueue`` keeps the canvases still to upload in a JSON file, so they are
uploaded after a restart or a lost network too.

Needs ``mikro`` (and dask, zarr); the import happens on use.
"""
import datetime
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MAX_ATTEMPTS = 3        # then the canvas is set aside as failed (the panel shows why)
KEEP_RECORDS = 200      # uploaded and failed entries remembered

STAGE_SPACE = "stage"   # per app: this microscope's stage, every run
# mikro-server shows a 3-channel dataset as one RGB layer only when its channels
# are labelled red/green/blue (core/logic/channels.py); ImSwitch's writer names
# them "<channel>_rgb_r" and so on.
RGB_LABELS = {"r": "red", "g": "green", "b": "blue",
              "red": "red", "green": "green", "blue": "blue"}


def _rgb_component(name: str) -> Optional[str]:
    low = name.lower()
    for key, label in (("rgb_r", "red"), ("rgb_g", "green"), ("rgb_b", "blue")):
        if low == key or low.endswith("_" + key):
            return label
    return RGB_LABELS.get(low)


@dataclass
class NgffCanvas:
    """One NGFF image on disk, opened lazily and squeezed to what mikro wants."""
    path: Path
    name: str
    levels: List[Any]                 # dask arrays, axes as in ``dims``
    dims: List[str]                   # e.g. ["c", "y", "x"]: singleton t and z dropped
    pixel_um: Dict[str, float]        # {"y": .., "x": ..}
    origin_um: Dict[str, float]       # top-left corner, {"y": .., "x": ..}
    labels: List[str] = field(default_factory=list)   # per channel

    @property
    def nbytes(self) -> int:
        return int(self.levels[0].nbytes)


def channel_labels(names: Sequence[str]) -> List[str]:
    """Channel names as mikro should see them: an RGB triple as red/green/blue."""
    names = [str(n) for n in names]
    rgb = [_rgb_component(n) for n in names]
    if len(names) == 3 and sorted(filter(None, rgb)) == ["blue", "green", "red"]:
        return rgb
    return names


def read_ngff(path) -> NgffCanvas:
    """Open the NGFF image group at *path* (NGFF 0.4 or 0.5) without reading pixels."""
    import dask.array as da
    import zarr

    path = Path(path)
    group = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    attrs = attrs.get("ome", attrs)   # 0.5 nests its metadata under "ome"
    multiscales = attrs["multiscales"][0]
    axes = [a["name"] for a in multiscales["axes"]]
    levels = [da.from_zarr(group[d["path"]]) for d in multiscales["datasets"]]
    base = levels[0].shape
    keep = [i for i, a in enumerate(axes) if not (a in ("t", "z") and base[i] == 1)]
    index = tuple(slice(None) if i in keep else 0 for i in range(len(axes)))
    transforms = {t["type"]: t for t in multiscales["datasets"][0]["coordinateTransformations"]}
    scale = transforms["scale"]["scale"]
    translation = transforms.get("translation", {}).get("translation", [0.0] * len(axes))
    names = [ch.get("label") or f"C{i}"
             for i, ch in enumerate(attrs.get("omero", {}).get("channels", []))]
    return NgffCanvas(path=path, name=multiscales.get("name") or path.name,
                      levels=[level[index] for level in levels], dims=[axes[i] for i in keep],
                      pixel_um={a: float(scale[axes.index(a)]) for a in ("y", "x")},
                      origin_um={a: float(translation[axes.index(a)]) for a in ("y", "x")},
                      labels=channel_labels(names))


def stage_space(mikro, name: str = "Stage"):
    """This microscope's stage as a µm space in mikro: created once, then reused."""
    from kanne.scalars import Unit
    from mikro import space_2d
    return space_2d(mikro, name, unit=Unit("micrometer"), reference=STAGE_SPACE)


def channel_anchors(array, dims: Sequence[str], labels: Sequence[str]) -> list:
    """Per-channel contrast (a value histogram) and name. *array* should be
    small: the coarsest level is enough for a display window."""
    from mikro.api.schema import CoordinateAnchorInput, LabelInput

    if "c" not in dims:
        return []
    anchors = CoordinateAnchorInput.histogram_anchors(array, axis="c", dims=list(dims))
    return [a.model_copy(update={"label": LabelInput(label=labels[i])}) if i < len(labels) else a
            for i, a in enumerate(anchors)]


def upload_ngff(mikro, canvas: NgffCanvas, *, name: Optional[str] = None, space=None):
    """Create a mikro dataset from *canvas*: level 0 as data, the coarser levels
    as its scales (recorded as NEAREST: NgffImage decimates by stride), the
    channels named and contrast-anchored. Registered in *space* at the canvas'
    top-left corner when given. Returns the dataset."""
    from mikro.api.schema import ScaleInput, ScaleMethod
    from mikro.scalars import ArrayLike

    dims, levels = canvas.dims, canvas.levels
    dataset = mikro.create_array_dataset(
        data=ArrayLike(levels[0], dims=dims), name=name or canvas.name, axes=dims,
        scales=[ScaleInput(level=i, array=ArrayLike(level, dims=dims),
                           scale_method=ScaleMethod.NEAREST)
                for i, level in enumerate(levels) if i > 0],
        anchors=channel_anchors(levels[-1].compute(), dims, canvas.labels))
    if space is not None:
        space.register(dataset, scale=canvas.pixel_um, **canvas.origin_um)
    return dataset


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class UploadQueue:
    """Canvases waiting for mikro, oldest first, persisted in *path* (JSON).

    {"pending": [{path, name, queuedAt, attempts, error}],
     "uploaded": [{path, name, datasetId, seconds, uploadedAt}],
     "failed": [...]}. Thread-safe; wait() wakes on add()."""

    def __init__(self, path):
        self.path = Path(path)
        self._changed = threading.Condition()
        self._state = {"pending": [], "uploaded": [], "failed": []}
        try:
            self._state.update(json.loads(self.path.read_text()))
        except (OSError, ValueError):
            pass

    def add(self, canvas_path, name: Optional[str] = None) -> bool:
        """Queue a canvas; False when it is queued or uploaded already."""
        canvas_path = str(canvas_path)
        with self._changed:
            if any(e["path"] == canvas_path for e in self._state["pending"] + self._state["uploaded"]):
                return False
            self._state["pending"].append({"path": canvas_path, "name": name, "queuedAt": _now(),
                                           "attempts": 0, "error": None})
            self._save()
            self._changed.notify_all()
        return True

    def next(self) -> Optional[dict]:
        with self._changed:
            return dict(self._state["pending"][0]) if self._state["pending"] else None

    def wait(self, timeout: float) -> None:
        with self._changed:
            self._changed.wait(timeout)

    def done(self, entry: dict, dataset_id, seconds: float) -> None:
        self._finish(entry, "uploaded", datasetId=str(dataset_id), seconds=round(seconds, 1),
                     uploadedAt=_now(), error=None)

    def failed(self, entry: dict, error: BaseException) -> None:
        """Count a failed attempt; after MAX_ATTEMPTS the canvas is set aside."""
        message = f"{type(error).__name__}: {error}"[:300]
        with self._changed:
            for pending in self._state["pending"]:
                if pending["path"] == entry["path"]:
                    pending["attempts"] += 1
                    pending["error"] = message
                    if pending["attempts"] < MAX_ATTEMPTS:
                        self._state["pending"].append(self._state["pending"].pop(
                            self._state["pending"].index(pending)))  # let the others go first
                        self._save()
                        return
        self._finish(entry, "failed", error=message, failedAt=_now())

    def snapshot(self) -> dict:
        with self._changed:
            return json.loads(json.dumps(self._state))

    def _finish(self, entry: dict, bucket: str, **fields) -> None:
        with self._changed:
            pending = [e for e in self._state["pending"] if e["path"] == entry["path"]]
            self._state["pending"] = [e for e in self._state["pending"] if e["path"] != entry["path"]]
            record = {**(pending[0] if pending else entry), **fields}
            self._state[bucket] = (self._state[bucket] + [record])[-KEEP_RECORDS:]
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._state, indent=1))
        os.replace(temporary, self.path)
