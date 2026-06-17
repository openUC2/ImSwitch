"""mm -> µm unit conversion test."""

from imswitch.imcontrol.model.labware import load_labware_from_dict


_MINIMAL = {
    "schemaVersion": 2,
    "version": 1,
    "namespace": "test",
    "metadata": {
        "displayName": "Test 1x1",
        "displayCategory": "wellPlate",
        "displayVolumeUnits": "µL",
        "tags": [],
    },
    "brand": {"brand": "Test", "brandId": [], "links": []},
    "parameters": {
        "format": "irregular",
        "isTiprack": False,
        "loadName": "test_1x1",
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
            "x": 10.0,
            "y": 20.0,
            "z": 1.5,
            "diameter": 6.0,
        }
    },
    "groups": [],
}


def test_mm_to_um_conversion():
    lab = load_labware_from_dict(_MINIMAL)
    assert lab.dimensions.x == 80000.0
    assert lab.dimensions.y == 60000.0
    assert lab.dimensions.z == 12000.0
    w = lab.get_well("A1")
    assert w.x == 10000.0
    assert w.y == 20000.0
    assert w.z == 1500.0
    assert w.geometry.shape == "circle"
    assert w.geometry.radius == 3000.0  # diameter/2 * 1000
    assert w.geometry.depth == 8000.0
    # Volume stays in µL — it isn't a length.
    assert w.geometry.totalLiquidVolume_uL == 100.0


def test_rectangular_conversion():
    data = dict(_MINIMAL)
    data = {**data, "wells": {
        "A1": {
            "shape": "rectangular",
            "depth": 5.0,
            "totalLiquidVolume": 50.0,
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "xDimension": 4.0,
            "yDimension": 3.0,
        }
    }}
    lab = load_labware_from_dict(data)
    w = lab.get_well("A1")
    assert w.geometry.shape == "rectangle"
    assert w.geometry.width == 4000.0
    assert w.geometry.height == 3000.0
