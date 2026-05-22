"""
OME-NGFF plate metadata sidecar writer.

When an experiment is bound to an Opentrons-style labware definition we emit a
JSON sidecar (``plate_metadata.json``) into the experiment directory describing
the plate layout per the OME-NGFF 0.4 plate spec
(https://ngff.openmicroscopy.org/0.4/#plate-md).

This is intentionally a sidecar rather than a true Zarr ``plate`` group:
ImSwitch writes one ``.ome.zarr`` per acquisition tile group, and refactoring
the on-disk layout into ``<plate>.ome.zarr/<row>/<col>/<acq>`` would be a
separate, larger change.  The sidecar is enough for downstream readers (Fractal,
napari-ome-zarr, OMERO importer prep scripts) to reconstruct the plate
relationship between the per-well stores.

TODO: In the future, if we want to support more complex plate relationships (e.g. multiple acquisitions per plate, or wells that don't fit a regular grid), we could
consider writing a full OME-NGFF plate group with links to the per-well stores instead of a sidecar.  But for now this simple approach is sufficient for the Opentrons-style plates we support.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def build_plate_metadata(
    plate_name: str,
    rows: Sequence[str],
    columns: Sequence[str],
    wells_used: Sequence[Tuple[str, str]],
    *,
    field_count: int = 1,
    acquisition_id: int = 0,
    acquisition_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return an OME-NGFF 0.4 ``plate`` attrs dict.

    Args:
        plate_name: Human-readable plate name.
        rows: All row labels declared by the plate (e.g. ``["A","B",...,"H"]``).
        columns: All column labels declared by the plate (e.g. ``["1",...,"12"]``).
        wells_used: Subset of wells that actually have data, as
            ``[(row, column), ...]``.  Order is preserved.
        field_count: Number of fields-of-view per well (informational).
        acquisition_id: Acquisition index.
        acquisition_name: Optional acquisition name.

    Returns:
        Dict ready to be JSON-serialised under the ``plate`` attrs key.
    """
    seen: set = set()
    well_entries: List[Dict[str, Any]] = []
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(columns)}
    for row, col in wells_used:
        key = (row, col)
        if key in seen:
            continue
        if row not in row_index or col not in col_index:
            # Skip wells that don't fit the declared rows/columns instead of
            # writing a malformed plate.
            continue
        seen.add(key)
        well_entries.append({
            "path": f"{row}/{col}",
            "rowIndex": row_index[row],
            "columnIndex": col_index[col],
        })

    plate: Dict[str, Any] = {
        "version": "0.4",
        "name": plate_name,
        "field_count": int(field_count),
        "rows": [{"name": r} for r in rows],
        "columns": [{"name": c} for c in columns],
        "wells": well_entries,
        "acquisitions": [
            {
                "id": int(acquisition_id),
                "name": acquisition_name or plate_name,
            }
        ],
    }
    return plate


def write_plate_metadata_sidecar(
    output_dir: str,
    plate_name: str,
    rows: Sequence[str],
    columns: Sequence[str],
    wells_used: Iterable[Tuple[str, str]],
    *,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "plate_metadata.json",
) -> Optional[str]:
    """Write a plate metadata sidecar JSON file.

    Args:
        output_dir: Directory to write the sidecar into. Created if missing.
        plate_name: Human-readable plate name (typically the labware loadName).
        rows: Row labels.
        columns: Column labels.
        wells_used: Iterable of (row, column) tuples actually scanned.
        extra: Optional ImSwitch-specific extension fields merged at the top
            level of the sidecar (e.g. labware loadName, condition labels).
        filename: Output filename inside ``output_dir``.

    Returns:
        Absolute path to the written file, or ``None`` if no wells were
        provided (sidecar is skipped to avoid emitting empty metadata).
    """
    wells_list = [(str(r), str(c)) for r, c in wells_used if r and c]
    if not wells_list:
        return None

    plate = build_plate_metadata(plate_name, rows, columns, wells_list)
    payload: Dict[str, Any] = {"plate": plate}
    if extra:
        payload.update(extra)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


__all__ = ["build_plate_metadata", "write_plate_metadata_sidecar"]
