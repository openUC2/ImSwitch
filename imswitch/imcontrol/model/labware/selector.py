"""Well-selection helpers.

Resolves a :class:`WellSelectionPattern` against a :class:`LabwareDefinition`
into an ordered, deduplicated list of :class:`WellPosition` (row-major).
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel

from .models import LabwareDefinition, WellPosition

_WELL_ID_RE = re.compile(r"^([A-Z]+)(\d+)$")


class WellSelectionPattern(BaseModel):
    """Composable selection pattern.  Empty pattern selects nothing.

    Multiple fields combine with set-union semantics.  Wells that don't exist
    on the labware are silently skipped (range patterns) or raise (explicit
    well/row/column patterns).
    """

    wells: Optional[List[str]] = None
    rows: Optional[List[str]] = None
    columns: Optional[List[int]] = None
    ranges: Optional[List[str]] = None
    all: bool = False


def _norm_well_id(raw: str) -> str:
    s = raw.strip().upper()
    if not _WELL_ID_RE.match(s):
        raise ValueError(f"Invalid well id: {raw!r}")
    return s


def _parse_range_token(raw: str) -> tuple[str, int, str, int]:
    """Parse ``"A1:C3"`` -> (row1, col1, row2, col2).  Order-tolerant."""
    s = raw.strip()
    if ":" not in s:
        raise ValueError(f"Invalid range token {raw!r} (expected 'A1:C3')")
    a, b = s.split(":", 1)
    a = a.strip().upper()
    b = b.strip().upper()
    ma = _WELL_ID_RE.match(a)
    mb = _WELL_ID_RE.match(b)
    if not ma or not mb:
        raise ValueError(f"Invalid range token {raw!r}")
    r1, c1 = ma.group(1), int(ma.group(2))
    r2, c2 = mb.group(1), int(mb.group(2))
    if r1 > r2:
        r1, r2 = r2, r1
    if c1 > c2:
        c1, c2 = c2, c1
    return r1, c1, r2, c2


def _row_letters_between(rows: List[str], r1: str, r2: str) -> List[str]:
    """Return the slice of ``rows`` between r1 and r2 (inclusive) keeping the
    labware's row ordering.  Skips rows outside the labware silently."""
    if r1 not in rows and r2 not in rows:
        # Bound the range using lexicographic comparison.
        return [r for r in rows if r1 <= r <= r2]
    if r1 not in rows:
        return [r for r in rows if r <= r2]
    if r2 not in rows:
        return [r for r in rows if r >= r1]
    i1 = rows.index(r1)
    i2 = rows.index(r2)
    return rows[i1 : i2 + 1]


def resolve_pattern(
    labware: LabwareDefinition,
    pattern: WellSelectionPattern,
) -> List[WellPosition]:
    """Resolve a pattern against a labware.

    Returns wells in row-major order (A1, A2, ..., B1, ...).  Deduplicated.
    """
    selected: set[str] = set()

    if pattern.all:
        selected.update(labware.wells.keys())

    if pattern.wells:
        for raw in pattern.wells:
            wid = _norm_well_id(raw)
            if wid not in labware.wells:
                raise ValueError(
                    f"Well {wid!r} not in labware {labware.load_name!r}"
                )
            selected.add(wid)

    if pattern.rows:
        for raw in pattern.rows:
            r = raw.strip().upper()
            if r not in labware.rows:
                raise ValueError(
                    f"Row {r!r} not in labware {labware.load_name!r}"
                )
            for c in labware.columns:
                wid = f"{r}{c}"
                if wid in labware.wells:
                    selected.add(wid)

    if pattern.columns:
        for c in pattern.columns:
            if c not in labware.columns:
                raise ValueError(
                    f"Column {c!r} not in labware {labware.load_name!r}"
                )
            for r in labware.rows:
                wid = f"{r}{c}"
                if wid in labware.wells:
                    selected.add(wid)

    if pattern.ranges:
        for raw in pattern.ranges:
            r1, c1, r2, c2 = _parse_range_token(raw)
            for r in _row_letters_between(labware.rows, r1, r2):
                for c in range(c1, c2 + 1):
                    wid = f"{r}{c}"
                    if wid in labware.wells:
                        selected.add(wid)

    # Row-major ordering (uses the labware's flat order).
    return [labware.wells[wid] for wid in labware.well_names_flat if wid in selected]


__all__ = ["WellSelectionPattern", "resolve_pattern"]
