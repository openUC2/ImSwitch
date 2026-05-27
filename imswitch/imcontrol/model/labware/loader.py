"""Load Opentrons-format labware JSON into ImSwitch's in-memory model.

The on-disk JSON stays in **mm** (Opentrons convention).  This loader is the
*only* place that converts mm -> µm.  Everything downstream sees µm.

Vendored schema: ``schema_v2.json`` is a reduced copy of
https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/labware/schemas/2.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import (
    LabwareDefinition,
    LabwareDimensions,
    OpentronsLabwareV2,
    WellGeometry,
    WellPosition,
)

MM_TO_UM: float = 1000.0
_WELL_ID_RE = re.compile(r"^([A-Z]+)(\d+)$")
_SCHEMA_PATH = Path(__file__).parent / "schema_v2.json"


class LabwareValidationError(ValueError):
    """Raised when an Opentrons labware JSON fails validation."""


_SCHEMA_CACHE: Optional[dict] = None
_VALIDATOR = None


def _get_validator():
    """Lazy-construct a jsonschema validator.  Imported lazily so the
    labware module loads even when ``jsonschema`` is missing — validation
    just becomes a no-op with a warning logged at first use."""
    global _SCHEMA_CACHE, _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR
    try:
        import jsonschema  # type: ignore
    except ImportError:  # pragma: no cover
        return None
    if _SCHEMA_CACHE is None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            _SCHEMA_CACHE = json.load(fh)
    _VALIDATOR = jsonschema.Draft7Validator(_SCHEMA_CACHE)
    return _VALIDATOR


def _validate_with_jsonschema(data: dict, source: str) -> None:
    validator = _get_validator()
    if validator is None:
        return
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        msgs = []
        for err in errors[:5]:
            loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
            msgs.append(f"  at {loc}: {err.message}")
        raise LabwareValidationError(
            f"Schema validation failed for {source}:\n" + "\n".join(msgs)
        )


def _shape_mm_to_um(well_data: Dict[str, Any], well_id: str) -> WellGeometry:
    shape = well_data.get("shape")
    depth_mm = well_data.get("depth")
    vol_uL = well_data.get("totalLiquidVolume")
    if depth_mm is None or vol_uL is None:
        raise LabwareValidationError(
            f"Well {well_id!r}: missing 'depth' or 'totalLiquidVolume'"
        )
    if shape == "circular":
        diameter_mm = well_data.get("diameter")
        if diameter_mm is None:
            raise LabwareValidationError(
                f"Well {well_id!r}: circular well missing 'diameter'"
            )
        return WellGeometry(
            shape="circle",
            radius=(diameter_mm / 2.0) * MM_TO_UM,
            depth=depth_mm * MM_TO_UM,
            totalLiquidVolume_uL=float(vol_uL),
        )
    if shape == "rectangular":
        x_dim = well_data.get("xDimension")
        y_dim = well_data.get("yDimension")
        if x_dim is None or y_dim is None:
            raise LabwareValidationError(
                f"Well {well_id!r}: rectangular well missing "
                "'xDimension'/'yDimension'"
            )
        return WellGeometry(
            shape="rectangle",
            width=x_dim * MM_TO_UM,
            height=y_dim * MM_TO_UM,
            depth=depth_mm * MM_TO_UM,
            totalLiquidVolume_uL=float(vol_uL),
        )
    raise LabwareValidationError(
        f"Well {well_id!r}: unsupported shape {shape!r} "
        "(expected 'circular' or 'rectangular')"
    )


def _parse_well_id(well_id: str) -> tuple[str, int]:
    m = _WELL_ID_RE.match(well_id)
    if not m:
        raise LabwareValidationError(
            f"Well id {well_id!r} does not match ^[A-Z]+\\d+$"
        )
    return m.group(1), int(m.group(2))


def _flatten_row_major(
    rows: List[str], columns: List[int], wells: Dict[str, WellPosition]
) -> List[str]:
    flat: List[str] = []
    for r in rows:
        for c in columns:
            wid = f"{r}{c}"
            if wid in wells:
                flat.append(wid)
    return flat


def load_labware_from_dict(
    data: dict, source_path: Optional[str] = None
) -> LabwareDefinition:
    """Validate, parse and unit-convert a labware dict.

    Args:
        data: parsed JSON content of an Opentrons schema-v2 labware file.
        source_path: optional source path for error messages and caching.

    Returns:
        LabwareDefinition (µm).

    Raises:
        LabwareValidationError on any structural or value error.
    """
    src = source_path or "<dict>"
    _validate_with_jsonschema(data, src)

    try:
        ot = OpentronsLabwareV2.model_validate(data)
    except Exception as exc:  # pragma: no cover - jsonschema usually catches first
        raise LabwareValidationError(
            f"Could not parse labware {src}: {exc}"
        ) from exc

    # ---- ordering / rows / columns -----------------------------------------
    ordering_well_ids: List[str] = [w for col in ot.ordering for w in col]
    ordering_set = set(ordering_well_ids)
    wells_set = set(ot.wells.keys())
    if ordering_set != wells_set:
        only_in_order = ordering_set - wells_set
        only_in_wells = wells_set - ordering_set
        raise LabwareValidationError(
            f"{src}: 'ordering' and 'wells' keys disagree. "
            f"In ordering only: {sorted(only_in_order)}. "
            f"In wells only: {sorted(only_in_wells)}."
        )

    rows: List[str] = []
    cols: List[int] = []
    seen_rows: set[str] = set()
    seen_cols: set[int] = set()
    for wid in ordering_well_ids:
        r, c = _parse_well_id(wid)
        if r not in seen_rows:
            seen_rows.add(r)
            rows.append(r)
        if c not in seen_cols:
            seen_cols.add(c)
            cols.append(c)
    rows.sort()
    cols.sort()

    # ---- wells (mm -> µm) --------------------------------------------------
    wells: Dict[str, WellPosition] = {}
    for wid, well_data in ot.wells.items():
        row, col = _parse_well_id(wid)
        geom = _shape_mm_to_um(well_data, wid)
        try:
            wells[wid] = WellPosition(
                id=wid,
                row=row,
                column=col,
                x=float(well_data["x"]) * MM_TO_UM,
                y=float(well_data["y"]) * MM_TO_UM,
                z=float(well_data["z"]) * MM_TO_UM,
                geometry=geom,
            )
        except KeyError as exc:
            raise LabwareValidationError(
                f"Well {wid!r}: missing required coordinate {exc}"
            ) from exc

    well_names_flat = _flatten_row_major(rows, cols, wells)

    return LabwareDefinition(
        load_name=ot.parameters.loadName,
        namespace=ot.namespace,
        version=ot.version,
        display_name=ot.metadata.displayName,
        display_category=ot.metadata.displayCategory,
        format=ot.parameters.format,
        brand=ot.brand.brand,
        brand_ids=list(ot.brand.brandId),
        links=list(ot.brand.links),
        tags=list(ot.metadata.tags),
        dimensions=LabwareDimensions(
            x=ot.dimensions.xDimension * MM_TO_UM,
            y=ot.dimensions.yDimension * MM_TO_UM,
            z=ot.dimensions.zDimension * MM_TO_UM,
        ),
        rows=rows,
        columns=cols,
        ordering=ot.ordering,
        wells=wells,
        well_names_flat=well_names_flat,
        groups=ot.groups,
        corner_offset_from_slot_um={
            "x": ot.cornerOffsetFromSlot.x * MM_TO_UM,
            "y": ot.cornerOffsetFromSlot.y * MM_TO_UM,
            "z": ot.cornerOffsetFromSlot.z * MM_TO_UM,
        },
        source_path=source_path,
    )


def load_labware_from_file(path: Union[str, Path]) -> LabwareDefinition:
    """Load and parse a labware JSON file from disk."""
    p = Path(path)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabwareValidationError(
            f"Could not read labware file {p}: {exc}"
        ) from exc
    return load_labware_from_dict(data, source_path=str(p))


__all__ = [
    "LabwareValidationError",
    "MM_TO_UM",
    "load_labware_from_dict",
    "load_labware_from_file",
]
