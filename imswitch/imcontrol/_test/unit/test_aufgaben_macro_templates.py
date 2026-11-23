"""Bundled macro templates are listed with their run parameters."""
from imswitch.imcontrol.controller.controllers.AufgabenController import _macro_entries
from imswitch.imcontrol.model import aufgaben


def test_bundled_macro_templates():
    templates = {m["name"]: m for m in _macro_entries(aufgaben.EXAMPLES_DIR)}
    assert {"autofocus_here_macro", "shutter_close_macro", "shutter_turret_macro",
            "bead_psf_stack_macro"} <= set(templates)
    assert templates["autofocus_here_macro"]["title"] == "Autofocus here"
    assert set(templates["autofocus_here_macro"]["parameters"]) == {"range_um", "step_um"}
    assert "tile_scan_focus_check" not in templates  # a job, not a macro


def test_bead_psf_macro_takes_one_stack_and_returns_to_focus(tmp_path):
    import zarr
    from imswitch.imcontrol.model.aufgaben.engine import Run
    from imswitch.imcontrol.model.aufgaben.simulated import SimulatedMicroscope
    from imswitch.imcontrol.model.aufgaben.storage import OmeStore

    job = aufgaben.examples()["bead_psf_stack_macro"]
    sim = SimulatedMicroscope()
    assert aufgaben.validate(job, capabilities=sim.capabilities()).ok
    run = Run(job, sim, parameters={"range_um": 2, "step_um": 0.5},
              sink=OmeStore(tmp_path / "data"), run_dir=tmp_path)
    assert run.run(confirmed=True) == "finished", run.message
    stack = zarr.open_group(str(tmp_path / "data" / "beads.ome.zarr"), mode="r")
    assert stack["0"].shape[:3] == (1, 1, 5)  # -1 to +1 µm around the focus
    scale = stack.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale[2] == 0.5
    moves = [c[1]["Z"] for c in sim.calls if c[0] == "move" and "Z" in c[1]]
    assert moves[-1] == moves[2] == 0.0  # back at the focus it started from
