#!/usr/bin/env python3
"""WP-05 — one region model, one region identifier.

build_scan_regions replaced generate_snake_tiles, whose `centerIndex` was
written three different ways: the area's string id, an integer index, and a
literal 0. Acquisition then resolved the focus map with
`centerIndex or areaName or wellId or "default"` — and 0 is falsy in Python, so
the first region always fell through to the server-global "manual" map, which
outlives the experiment and the sample. That is the out-of-focus scan.

Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.controller.controllers.ExperimentController import (  # noqa: E402
    ExperimentController,
)
from imswitch.imcontrol.controller.controllers.experiment_controller.models import (  # noqa: E402
    CenterPosition,
    Experiment,
    NeighborPoint,
    ParameterValue,
    Point,
    ScanArea,
    ScanBounds,
    ScanPosition,
)

PARAMS = ParameterValue(
    timeLapsePeriod=0.0, numberOfImages=1, autoFocus=False,
    autoFocusMin=-1.0, autoFocusMax=1.0, autoFocusStepSize=1.0,
    zStack=False, zStackMin=0.0, zStackMax=0.0,
)


def _builder():
    """build_scan_regions + helpers bound to a bare stub."""
    stub = types.SimpleNamespace(
        _logger=types.SimpleNamespace(info=lambda *a, **k: None,
                                      warning=lambda *a, **k: None),
        mStage=types.SimpleNamespace(getPosition=lambda: {"X": 7.0, "Y": 8.0, "Z": 9.0}),
    )
    stub.build_scan_regions = types.MethodType(
        ExperimentController.build_scan_regions, stub
    )
    # The rest are @staticmethod — attach them as plain functions.
    for name in ("_build_region_meta", "_region_fovs", "regions_to_areas"):
        setattr(stub, name, getattr(ExperimentController, name))
    return stub


def _area(area_id, n=3, z=None):
    return ScanArea(
        areaId=area_id,
        areaName=f"Name {area_id}",
        centerPosition=CenterPosition(x=0.0, y=0.0, z=z),
        bounds=ScanBounds(minX=0, maxX=n, minY=0, maxY=n, width=n, height=n),
        positions=[
            ScanPosition(index=i, x=float(i), y=0.0, z=z, iX=i, iY=0) for i in range(n)
        ],
    )


def test_region_id_is_the_area_id_on_every_fov():
    regions, meta = _builder().build_scan_regions(
        Experiment(name="e", parameterValue=PARAMS, scanAreas=[_area("area_0"), _area("area_1")])
    )
    assert [fov["region_id"] for region in regions for fov in region] == (
        ["area_0"] * 3 + ["area_1"] * 3
    )
    assert set(meta) == {"area_0", "area_1"}
    assert meta["area_0"]["areaName"] == "Name area_0"


def test_region_id_is_always_a_string_and_never_falsy():
    """The literal 0 in the old fallback branch is what broke the lookup."""
    experiment = Experiment(
        name="e", parameterValue=PARAMS,
        pointList=[Point(name="p0", x=0.0, y=0.0), Point(name="p1", x=1.0, y=1.0)],
    )
    regions, _ = _builder().build_scan_regions(experiment)
    ids = [region[0]["region_id"] for region in regions]
    assert ids == ["area_0", "area_1"]
    assert all(isinstance(i, str) and i for i in ids)

    # ... and for the no-coordinates fallback, which used to emit 0.
    regions, meta = _builder().build_scan_regions(
        Experiment(name="e", parameterValue=PARAMS)
    )
    assert regions[0][0]["region_id"] == "current"
    assert (regions[0][0]["x"], regions[0][0]["y"]) == (7.0, 8.0)
    assert meta["current"]["areaName"] == "Current Position"


def test_region_metadata_is_not_copied_onto_every_fov():
    regions, _ = _builder().build_scan_regions(
        Experiment(name="e", parameterValue=PARAMS, scanAreas=[_area("area_0")])
    )
    assert set(regions[0][0]) == {"iterator", "region_id", "iX", "iY", "x", "y", "z"}


def test_z_is_homogeneous_across_a_region():
    """Squid's rule: a region has Z on every FOV or on none."""
    stub = _builder()

    with_z = stub.build_scan_regions(
        Experiment(name="e", parameterValue=PARAMS, scanAreas=[_area("a", z=12.5)])
    )[0][0]
    assert [fov["z"] for fov in with_z] == [12.5, 12.5, 12.5]

    without_z = stub.build_scan_regions(
        Experiment(name="e", parameterValue=PARAMS, scanAreas=[_area("a")])
    )[0][0]
    assert all(fov["z"] is None for fov in without_z)

    # One FOV missing Z demotes the whole region rather than mixing the two.
    mixed = _area("a", z=3.0)
    mixed.positions[1].z = None
    mixed.centerPosition.z = None
    partial = stub.build_scan_regions(
        Experiment(name="e", parameterValue=PARAMS, scanAreas=[mixed])
    )[0][0]
    assert all(fov["z"] is None for fov in partial)


def test_z_equals_zero_is_a_real_z():
    """The old code treated z == 0.0 as 'not given'."""
    regions, _ = _builder().build_scan_regions(
        Experiment(name="e", parameterValue=PARAMS, scanAreas=[_area("a", z=0.0)])
    )
    assert [fov["z"] for fov in regions[0]] == [0.0, 0.0, 0.0]


def test_empty_region_raises_naming_itself():
    stub = _builder()
    try:
        stub._region_fovs("area_3", [])
    except ValueError as err:
        assert "area_3" in str(err)
    else:
        raise AssertionError("an empty region must be rejected, not accepted")


def test_bad_coordinate_raises_before_anything_is_stored():
    stub = _builder()
    try:
        stub._region_fovs("area_9", [{"x": 1.0, "y": 2.0}, {"x": None, "y": 2.0}])
    except ValueError as err:
        assert "area_9" in str(err) and "1" in str(err)  # names region and index
    else:
        raise AssertionError("a malformed coordinate must be rejected")


def test_regions_to_areas_matches_the_region_ids():
    """The focus map is stored under these ids and looked up by region_id."""
    stub = _builder()
    experiment = Experiment(
        name="e", parameterValue=PARAMS, scanAreas=[_area("area_0"), _area("area_1")]
    )
    regions, meta = stub.build_scan_regions(experiment)
    areas = stub.regions_to_areas(regions, meta)

    assert [a["areaId"] for a in areas] == ["area_0", "area_1"]
    assert {a["areaId"] for a in areas} == {r[0]["region_id"] for r in regions}
    assert areas[0]["bounds"] == {"minX": 0.0, "maxX": 2.0, "minY": 0.0, "maxY": 0.0}


def test_neighbour_points_become_one_region_per_center():
    experiment = Experiment(
        name="e", parameterValue=PARAMS,
        pointList=[Point(
            name="p0", x=0.0, y=0.0,
            neighborPointList=[
                NeighborPoint(x=0.0, y=0.0, iX=0, iY=0),
                NeighborPoint(x=1.0, y=0.0, iX=1, iY=0),
            ],
        )],
    )
    regions, meta = _builder().build_scan_regions(experiment)
    assert len(regions) == 1 and len(regions[0]) == 2
    assert all(fov["region_id"] == "area_0" for fov in regions[0])
    assert meta["area_0"]["areaName"] == "p0"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
