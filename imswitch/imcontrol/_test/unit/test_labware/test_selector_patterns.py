"""WellSelectionPattern tests."""

import pytest

from imswitch.imcontrol.model.labware import (
    LabwareManager,
    WellSelectionPattern,
    resolve_pattern,
)


@pytest.fixture(scope="module")
def labware_96():
    mgr = LabwareManager()
    return mgr.get("corning_96_wellplate_360ul_flat")


def _ids(wells):
    return [w.id for w in wells]


def test_explicit_wells(labware_96):
    out = resolve_pattern(labware_96, WellSelectionPattern(wells=["A1", "B5"]))
    assert _ids(out) == ["A1", "B5"]


def test_lowercase_input(labware_96):
    out = resolve_pattern(labware_96, WellSelectionPattern(wells=["a1"]))
    assert _ids(out) == ["A1"]


def test_rows(labware_96):
    out = resolve_pattern(labware_96, WellSelectionPattern(rows=["A"]))
    assert _ids(out) == [f"A{c}" for c in range(1, 13)]


def test_columns(labware_96):
    out = resolve_pattern(labware_96, WellSelectionPattern(columns=[1, 12]))
    expected = [f"{r}{c}" for r in "ABCDEFGH" for c in (1, 12)]
    assert sorted(_ids(out)) == sorted(expected)


def test_range_full_plate(labware_96):
    out = resolve_pattern(labware_96, WellSelectionPattern(ranges=["A1:H12"]))
    assert len(out) == 96
    # Row-major ordering check.
    assert _ids(out)[:3] == ["A1", "A2", "A3"]


def test_range_single_well(labware_96):
    out = resolve_pattern(labware_96, WellSelectionPattern(ranges=["A1:A1"]))
    assert _ids(out) == ["A1"]


def test_range_reversed_endpoints(labware_96):
    a = resolve_pattern(labware_96, WellSelectionPattern(ranges=["A1:C3"]))
    b = resolve_pattern(labware_96, WellSelectionPattern(ranges=["C3:A1"]))
    assert _ids(a) == _ids(b)


def test_all(labware_96):
    out = resolve_pattern(labware_96, WellSelectionPattern(all=True))
    assert len(out) == 96


def test_dedup_and_combine(labware_96):
    out = resolve_pattern(
        labware_96,
        WellSelectionPattern(rows=["A"], columns=[1], wells=["A1"]),
    )
    # A row (12) ∪ col 1 (8) − A1 overlap = 19
    assert len(out) == 19


def test_malformed_range_raises(labware_96):
    with pytest.raises(ValueError):
        resolve_pattern(labware_96, WellSelectionPattern(ranges=["A1-C3"]))


def test_unknown_well_raises(labware_96):
    with pytest.raises(ValueError):
        resolve_pattern(labware_96, WellSelectionPattern(wells=["Z99"]))


def test_range_outside_plate_silently_skipped(labware_96):
    # 96-well plate: rows go A..H. Z is past the end and should be silently
    # ignored.
    out = resolve_pattern(labware_96, WellSelectionPattern(ranges=["A1:Z12"]))
    assert len(out) == 96
