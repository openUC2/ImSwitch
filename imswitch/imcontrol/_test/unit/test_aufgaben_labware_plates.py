"""ImSwitch labware as useq plates: every well lands where the labware says."""
import pytest
import useq

from imswitch.imcontrol.controller.controllers.aufgaben.host import register_labware, useq_plate
from imswitch.imcontrol.model.labware import LabwareManager


@pytest.fixture(scope="module")
def manager():
    return LabwareManager()


def test_regular_plates_match_their_labware_well_by_well(manager):
    names = register_labware(manager)
    assert {"corning_96_wellplate_360ul_flat", "corning_384_wellplate_112ul_flat"} <= set(names)
    for name in names:
        lab = manager.get_with_offset(name, 1000.0, -2000.0)   # plate origin on the stage
        a1 = lab.wells[f"{lab.rows[0]}{lab.columns[0]}"]
        plan = useq.WellPlatePlan(plate=name, a1_center_xy=(a1.x, a1.y),
                                  selected_wells=slice(None))
        positions = {p.name: (p.x, p.y) for p in plan.selected_well_positions}
        for well_name, well in lab.wells.items():
            assert positions[well_name] == pytest.approx((well.x, well.y), abs=0.01), \
                (name, well_name)


def test_irregular_labware_is_not_a_plate(manager):
    irregular = [s["load_name"] for s in manager.list_summaries()
                 if useq_plate(manager.get(s["load_name"])) is None]
    assert all(name not in useq.registered_well_plate_keys() for name in irregular)
