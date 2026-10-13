"""OME-XML for single-plane TIFFs (experiment tiles, snaps).

Built by the shared ``ome_xml`` of the OME writer, so every TIFF ImSwitch writes
carries the same metadata model: positions in µm (top-left pixel corner),
exposure in ms, physical pixel size, channel name.
"""
from typing import Any, Dict

from imswitch.imcontrol.model.aufgaben.storage import ImageSpec, ome_xml


def _first(metadata: Dict[str, Any], *keys, default=None):
    return next((metadata[k] for k in keys if metadata.get(k) is not None), default)


def build_ome_metadata_from_dict(metadata: Dict[str, Any]) -> bytes:
    """OME-XML for one plane described by an ImSwitch metadata dict.

    Required: ``height``, ``width``. Optional: ``dtype``, ``pixel_size``/``pixelSizeUm``
    (µm), ``x``/``y``/``z`` (stage position of the field centre, µm),
    ``channel_name``/``illuminationChannel``, ``exposure_time_ms``/``exposure`` and
    ``rgb`` (an interleaved RGB plane).
    """
    h, w = int(metadata["height"]), int(metadata["width"])
    px = float(_first(metadata, "pixel_size", "pixelSizeUm", "pixel_size_um", default=1.0))
    x, y, z = (float(_first(metadata, k, default=0.0)) for k in ("x", "y", "z"))
    channel = str(_first(metadata, "channel_name", "illuminationChannel", "channel",
                         default="C0"))
    exposure = _first(metadata, "exposure_time_ms", "exposure", "exposureTime")
    rgb = bool(metadata.get("rgb"))
    spec = ImageSpec(shape=(1, 3 if rgb else 1, 1, h, w),
                     dtype=str(metadata.get("dtype", "uint16")),
                     pixel_size_um=px, origin_um=(x - w * px / 2, y - h * px / 2, z),
                     channel_names=[channel] * (3 if rgb else 1),
                     exposure_ms=[float(exposure)] if exposure is not None else [],
                     name=str(metadata.get("name") or channel))
    return ome_xml(spec, rgb=rgb).encode("utf-8")   # Bio-Formats needs the µ as UTF-8
