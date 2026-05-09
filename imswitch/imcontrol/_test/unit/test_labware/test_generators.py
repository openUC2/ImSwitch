"""Generator tests — sanity-check geometry of generated SBS plates."""

from imswitch.imcontrol.model.labware import (
    generate_sbs_wellplate,
    load_labware_from_dict,
)


def test_corning_96_geometry():
    data = generate_sbs_wellplate(
        rows=8, cols=12,
        well_diameter_mm=6.86,
        well_depth_mm=10.67,
        total_volume_uL=360.0,
        well_spacing_x_mm=9.0,
        well_spacing_y_mm=9.0,
        a1_x_offset_mm=14.38,
        a1_y_offset_mm=11.24,
        load_name="corning_96_wellplate_360ul_flat",
        display_name="Corning 96",
        brand="Corning",
    )
    lab = load_labware_from_dict(data)
    assert len(lab.wells) == 96
    assert lab.format == "96Standard"
    # A1 -> back-left in Opentrons convention.
    a1 = lab.get_well("A1")
    h12 = lab.get_well("H12")
    # A1 at the larger y; H12 at the smaller y.
    assert a1.y > h12.y
    # 9mm spacing checks (µm).
    a2 = lab.get_well("A2")
    assert abs((a2.x - a1.x) - 9000.0) < 1e-6
    b1 = lab.get_well("B1")
    assert abs((a1.y - b1.y) - 9000.0) < 1e-6


def test_384_format():
    data = generate_sbs_wellplate(
        rows=16, cols=24,
        well_diameter_mm=3.30,
        well_depth_mm=11.43,
        total_volume_uL=112.0,
        well_spacing_x_mm=4.5,
        well_spacing_y_mm=4.5,
        a1_x_offset_mm=12.13,
        a1_y_offset_mm=8.99,
        load_name="corning_384_wellplate_112ul_flat",
        brand="Corning",
    )
    lab = load_labware_from_dict(data)
    assert lab.format == "384Standard"
    assert len(lab.wells) == 384
    # Row letters: A..P
    assert lab.rows[0] == "A"
    assert lab.rows[-1] == "P"
