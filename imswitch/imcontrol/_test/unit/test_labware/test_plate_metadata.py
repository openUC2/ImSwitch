"""Tests for the OME-NGFF plate metadata sidecar."""

import json
from pathlib import Path

import pytest

from imswitch.imcontrol.model.io.ome_writers.plate_metadata import (
    build_plate_metadata,
    write_plate_metadata_sidecar,
)


def test_build_plate_metadata_basic():
    plate = build_plate_metadata(
        plate_name="corning_96_wellplate_360ul_flat",
        rows=list("ABCDEFGH"),
        columns=[str(c) for c in range(1, 13)],
        wells_used=[("A", "1"), ("A", "2"), ("B", "1"), ("A", "1")],  # dup
    )
    assert plate["version"] == "0.4"
    assert plate["name"] == "corning_96_wellplate_360ul_flat"
    assert len(plate["rows"]) == 8
    assert len(plate["columns"]) == 12
    paths = [w["path"] for w in plate["wells"]]
    # Dedup preserved order, A1 appears once
    assert paths == ["A/1", "A/2", "B/1"]
    assert plate["wells"][0]["rowIndex"] == 0 and plate["wells"][0]["columnIndex"] == 0
    assert plate["wells"][2]["rowIndex"] == 1 and plate["wells"][2]["columnIndex"] == 0
    assert plate["acquisitions"][0]["id"] == 0


def test_build_plate_metadata_skips_unknown_wells():
    plate = build_plate_metadata(
        plate_name="x",
        rows=["A", "B"],
        columns=["1", "2"],
        wells_used=[("A", "1"), ("Z", "9"), ("B", "2")],
    )
    paths = [w["path"] for w in plate["wells"]]
    assert paths == ["A/1", "B/2"]


def test_write_sidecar_skipped_when_no_wells(tmp_path: Path):
    out = write_plate_metadata_sidecar(
        output_dir=str(tmp_path),
        plate_name="x",
        rows=["A"],
        columns=["1"],
        wells_used=[],
    )
    assert out is None
    assert not list(tmp_path.iterdir())


def test_write_sidecar_emits_json(tmp_path: Path):
    out = write_plate_metadata_sidecar(
        output_dir=str(tmp_path),
        plate_name="my_plate",
        rows=["A", "B"],
        columns=["1", "2", "3"],
        wells_used=[("A", "1"), ("B", "3")],
        extra={"imswitch_labware": {"loadName": "my_plate", "conditionLabels": {"A1": "ctrl"}}},
    )
    assert out is not None
    payload = json.loads(Path(out).read_text())
    assert payload["plate"]["name"] == "my_plate"
    assert payload["plate"]["wells"][0]["path"] == "A/1"
    assert payload["imswitch_labware"]["conditionLabels"] == {"A1": "ctrl"}
