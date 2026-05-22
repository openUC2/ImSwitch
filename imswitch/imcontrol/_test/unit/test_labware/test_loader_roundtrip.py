"""Round-trip tests for shipped labware definitions."""

from pathlib import Path

import pytest

from imswitch.imcontrol.model.labware import (
    LabwareManager,
    load_labware_from_file,
)


DEFS_ROOT = (
    Path(__file__).resolve().parents[3] / "model" / "labware" / "definitions"
)


def _all_definition_files():
    return sorted(p for p in DEFS_ROOT.rglob("*.json") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", _all_definition_files(), ids=lambda p: p.parent.name)
def test_definition_loads(path):
    lab = load_labware_from_file(path)

    # Round-trip integrity.
    flat = sum(len(c) for c in lab.ordering)
    assert len(lab.well_names_flat) == flat
    assert set(lab.well_names_flat) == set(lab.wells.keys())

    # All well coordinates positive (mm -> µm conversion sanity) and inside
    # the plate footprint.
    for wid, well in lab.wells.items():
        assert well.x >= 0, f"{wid}: x={well.x}"
        assert well.y >= 0, f"{wid}: y={well.y}"
        assert well.x <= lab.dimensions.x + 1, f"{wid}: x={well.x} > plate {lab.dimensions.x}"
        assert well.y <= lab.dimensions.y + 1, f"{wid}: y={well.y} > plate {lab.dimensions.y}"
        assert well.geometry.depth > 0


def test_labware_manager_loads_all():
    mgr = LabwareManager()
    names = mgr.list_load_names()
    assert "corning_96_wellplate_360ul_flat" in names
    assert "greiner_96_wellplate_650ul_uclear" in names
    assert "ibidi_8well_chambered_coverslip" in names
    assert "slide_4x_histosample_heidstar" in names
    summaries = mgr.list_summaries()
    assert all("display_name" in s for s in summaries)
