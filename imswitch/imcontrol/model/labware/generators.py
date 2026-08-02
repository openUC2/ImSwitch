"""Generators for SBS-standard plates -> Opentrons-format JSON dicts (mm).

Output is **mm**, exactly as Opentrons stores on disk.  Round-tripping through
the loader exercises the mm->µm conversion path on every startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional


def _row_letters(n: int) -> Iterator[str]:
    """A, B, ..., Z, AA, AB, ... (handles 1536-well 32-row plates)."""
    i = 0
    yielded = 0
    while yielded < n:
        # Convert i to base-26 with letters A-Z (1-indexed Excel-style).
        s = ""
        x = i + 1
        while x > 0:
            x, r = divmod(x - 1, 26)
            s = chr(65 + r) + s
        yield s
        yielded += 1
        i += 1


def generate_sbs_wellplate(
    rows: int,
    cols: int,
    well_diameter_mm: float,
    well_depth_mm: float,
    total_volume_uL: float,
    well_spacing_x_mm: float,
    well_spacing_y_mm: float,
    a1_x_offset_mm: float,
    a1_y_offset_mm: float,
    plate_x_mm: float = 127.76,
    plate_y_mm: float = 85.48,
    plate_z_mm: float = 14.35,
    load_name: str = "",
    display_name: str = "",
    brand: str = "Generic",
    namespace: str = "openuc2",
    version: int = 1,
    tags: Optional[List[str]] = None,
    well_shape: str = "circular",
    well_x_dim_mm: Optional[float] = None,
    well_y_dim_mm: Optional[float] = None,
    format_override: Optional[str] = None,
) -> dict:
    """Build an Opentrons-format labware dict for an SBS-style wellplate.

    Coordinate convention (Opentrons schema v2): well ``x`` and ``y`` are the
    distance from the labware's left-front-bottom corner to the well's
    center-bottom.  ``a1_x_offset_mm`` and ``a1_y_offset_mm`` are the
    distances from the plate's *left* and *front* edges to the A1 well center.
    Rows progress from front to back, so ``A`` is at the smallest ``y`` only
    when ``a1_y_offset_mm`` is small; row B is at ``y = a1_y_offset_mm +
    spacing_y``.

    For SBS plates Opentrons usually puts A1 at the **back-left** corner, so
    A is at the *largest* y.  Mirror that here:
        ``well.y = plate_y_mm - a1_y_offset_mm - row_index * spacing_y``

    Note: when comparing against a real Opentrons definition use the same
    a1_y_offset_mm value the upstream definition records (it is measured from
    the front edge in their convention).  Caller supplies the value matching
    the datasheet they're mirroring.
    """
    if not load_name:
        raise ValueError("load_name required")
    tags = list(tags or [])
    row_ids = list(_row_letters(rows))
    col_ids = list(range(1, cols + 1))

    ordering: List[List[str]] = []
    wells: dict = {}
    for ci, c in enumerate(col_ids):
        col_well_ids: List[str] = []
        for ri, r in enumerate(row_ids):
            wid = f"{r}{c}"
            col_well_ids.append(wid)
            x_mm = a1_x_offset_mm + ci * well_spacing_x_mm
            # Opentrons SBS convention: A1 at back-left, y decreases from A
            # towards the last row (front of plate).
            y_mm = plate_y_mm - a1_y_offset_mm - ri * well_spacing_y_mm
            well_obj = {
                "shape": well_shape,
                "depth": well_depth_mm,
                "totalLiquidVolume": total_volume_uL,
                "x": round(x_mm, 4),
                "y": round(y_mm, 4),
                "z": 0.0,
            }
            if well_shape == "circular":
                well_obj["diameter"] = well_diameter_mm
            elif well_shape == "rectangular":
                if well_x_dim_mm is None or well_y_dim_mm is None:
                    raise ValueError(
                        "rectangular wells require well_x_dim_mm/well_y_dim_mm"
                    )
                well_obj["xDimension"] = well_x_dim_mm
                well_obj["yDimension"] = well_y_dim_mm
            else:
                raise ValueError(f"Unsupported well_shape {well_shape!r}")
            wells[wid] = well_obj
        ordering.append(col_well_ids)

    if format_override is not None:
        fmt = format_override
    elif rows == 8 and cols == 12:
        fmt = "96Standard"
    elif rows == 16 and cols == 24:
        fmt = "384Standard"
    else:
        fmt = "irregular"

    return {
        "schemaVersion": 2,
        "version": version,
        "namespace": namespace,
        "metadata": {
            "displayName": display_name or load_name,
            "displayCategory": "wellPlate",
            "displayVolumeUnits": "µL",
            "tags": tags,
        },
        "brand": {"brand": brand, "brandId": [], "links": []},
        "parameters": {
            "format": fmt,
            "isTiprack": False,
            "loadName": load_name,
            "isMagneticModuleCompatible": False,
            "quirks": [],
            "isDeckSlotCompatible": True,
        },
        "ordering": ordering,
        "cornerOffsetFromSlot": {"x": 0.0, "y": 0.0, "z": 0.0},
        "dimensions": {
            "xDimension": plate_x_mm,
            "yDimension": plate_y_mm,
            "zDimension": plate_z_mm,
        },
        "wells": wells,
        "groups": [
            {
                "wells": [f"{r}{c}" for c in col_ids for r in row_ids],
                "metadata": {"displayCategory": "wellPlate"},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Built-in catalog generation (`python -m imswitch.imcontrol.model.labware.generators`)
# ---------------------------------------------------------------------------


_BUILTIN_SPECS = [
    # Corning 96-well 360 µL flat — SBS standard.
    dict(
        rows=8, cols=12,
        well_diameter_mm=6.86,
        well_depth_mm=10.67,
        total_volume_uL=360.0,
        well_spacing_x_mm=9.0,
        well_spacing_y_mm=9.0,
        a1_x_offset_mm=14.38,
        a1_y_offset_mm=11.24,
        load_name="corning_96_wellplate_360ul_flat",
        display_name="Corning 96 Well Plate 360 µL Flat",
        brand="Corning",
        tags=["SBS", "wellPlate", "_generated"],
    ),
    # Corning 384-well 112 µL flat.
    dict(
        rows=16, cols=24,
        well_diameter_mm=3.30,
        well_depth_mm=11.43,
        total_volume_uL=112.0,
        well_spacing_x_mm=4.5,
        well_spacing_y_mm=4.5,
        a1_x_offset_mm=12.13,
        a1_y_offset_mm=8.99,
        load_name="corning_384_wellplate_112ul_flat",
        display_name="Corning 384 Well Plate 112 µL Flat",
        brand="Corning",
        tags=["SBS", "wellPlate", "_generated"],
    ),
    # Greiner Bio-One 96-well µClear 650 µL — preferred for cellpainting/colony work.
    dict(
        rows=8, cols=12,
        well_diameter_mm=6.96,
        well_depth_mm=11.0,
        total_volume_uL=650.0,
        well_spacing_x_mm=9.0,
        well_spacing_y_mm=9.0,
        a1_x_offset_mm=14.38,
        a1_y_offset_mm=11.24,
        load_name="greiner_96_wellplate_650ul_uclear",
        display_name="Greiner 96 Well µClear 650 µL Flat",
        brand="Greiner Bio-One",
        tags=["SBS", "wellPlate", "cell-painting", "colony-counting",
              "hematopoietic-stem-cells", "_generated"],
    ),
    # Corning 6-well 16.8 mL flat — SBS standard, 2x3 layout.
    dict(
        rows=2, cols=3,
        well_diameter_mm=34.8,
        well_depth_mm=17.4,
        total_volume_uL=16800.0,
        well_spacing_x_mm=39.12,
        well_spacing_y_mm=39.12,
        a1_x_offset_mm=24.55,
        a1_y_offset_mm=23.16,
        load_name="corning_6_wellplate_16.8ml_flat",
        display_name="Corning 6 Well Plate 16.8 mL Flat",
        brand="Corning",
        format_override="6Standard",
        tags=["SBS", "wellPlate", "_generated"],
    ),
    # Corning 12-well 6.9 mL flat — 3x4 layout.
    dict(
        rows=3, cols=4,
        well_diameter_mm=22.11,
        well_depth_mm=17.53,
        total_volume_uL=6900.0,
        well_spacing_x_mm=26.01,
        well_spacing_y_mm=26.01,
        a1_x_offset_mm=24.94,
        a1_y_offset_mm=16.79,
        load_name="corning_12_wellplate_6.9ml_flat",
        display_name="Corning 12 Well Plate 6.9 mL Flat",
        brand="Corning",
        format_override="12Standard",
        tags=["SBS", "wellPlate", "_generated"],
    ),
    # Corning 24-well 3.4 mL flat — 4x6 layout.
    dict(
        rows=4, cols=6,
        well_diameter_mm=16.26,
        well_depth_mm=17.4,
        total_volume_uL=3400.0,
        well_spacing_x_mm=19.3,
        well_spacing_y_mm=19.3,
        a1_x_offset_mm=17.48,
        a1_y_offset_mm=13.67,
        load_name="corning_24_wellplate_3.4ml_flat",
        display_name="Corning 24 Well Plate 3.4 mL Flat",
        brand="Corning",
        format_override="24Standard",
        tags=["SBS", "wellPlate", "_generated"],
    ),
    # Corning 48-well 1.6 mL flat — 6x8 layout.
    dict(
        rows=6, cols=8,
        well_diameter_mm=11.56,
        well_depth_mm=17.4,
        total_volume_uL=1600.0,
        well_spacing_x_mm=13.0,
        well_spacing_y_mm=13.0,
        a1_x_offset_mm=18.16,
        a1_y_offset_mm=10.08,
        load_name="corning_48_wellplate_1.6ml_flat",
        display_name="Corning 48 Well Plate 1.6 mL Flat",
        brand="Corning",
        format_override="48Standard",
        tags=["SBS", "wellPlate", "_generated"],
    ),
]


def _write(definitions_root: Path, spec: dict) -> Path:
    data = generate_sbs_wellplate(**spec)
    out_dir = definitions_root / spec["load_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data['version']}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return out_path


def regenerate_builtins(definitions_root: Optional[Path] = None) -> List[Path]:
    """Regenerate the built-in SBS plate JSONs into ``definitions/openuc2/``."""
    if definitions_root is None:
        definitions_root = Path(__file__).parent / "definitions" / "openuc2"
    written: List[Path] = []
    for spec in _BUILTIN_SPECS:
        written.append(_write(definitions_root, spec))
    return written


if __name__ == "__main__":  # pragma: no cover
    paths = regenerate_builtins()
    for p in paths:
        print(f"Wrote {p}")


__all__ = ["generate_sbs_wellplate", "regenerate_builtins"]
