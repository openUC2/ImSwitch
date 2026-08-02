"""Validation error tests."""

import pytest

from imswitch.imcontrol.model.labware import (
    LabwareValidationError,
    load_labware_from_dict,
)


def _base():
    return {
        "schemaVersion": 2,
        "version": 1,
        "namespace": "test",
        "metadata": {
            "displayName": "T",
            "displayCategory": "wellPlate",
            "displayVolumeUnits": "µL",
            "tags": [],
        },
        "brand": {"brand": "Test", "brandId": [], "links": []},
        "parameters": {
            "format": "irregular",
            "isTiprack": False,
            "loadName": "t",
            "isMagneticModuleCompatible": False,
            "quirks": [],
            "isDeckSlotCompatible": True,
        },
        "ordering": [["A1"]],
        "cornerOffsetFromSlot": {"x": 0, "y": 0, "z": 0},
        "dimensions": {"xDimension": 80.0, "yDimension": 60.0, "zDimension": 12.0},
        "wells": {
            "A1": {
                "shape": "circular",
                "depth": 8.0,
                "totalLiquidVolume": 100.0,
                "x": 10.0, "y": 20.0, "z": 1.5,
                "diameter": 6.0,
            }
        },
        "groups": [],
    }


def test_ordering_wells_disagree():
    data = _base()
    data["ordering"] = [["A1", "B1"]]  # B1 missing from wells
    with pytest.raises(LabwareValidationError, match="ordering.*wells"):
        load_labware_from_dict(data)


def test_unsupported_shape():
    data = _base()
    data["wells"]["A1"]["shape"] = "weird"
    with pytest.raises(LabwareValidationError):
        load_labware_from_dict(data)


def test_circular_well_missing_diameter():
    data = _base()
    data["wells"]["A1"].pop("diameter")
    with pytest.raises(LabwareValidationError):
        load_labware_from_dict(data)


def test_invalid_well_id():
    data = _base()
    data["ordering"] = [["1A"]]
    data["wells"] = {"1A": data["wells"]["A1"]}
    with pytest.raises(LabwareValidationError):
        load_labware_from_dict(data)
