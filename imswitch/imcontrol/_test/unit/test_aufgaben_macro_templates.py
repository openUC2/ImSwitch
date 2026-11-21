"""Bundled macro templates are listed with their run parameters."""
from imswitch.imcontrol.controller.controllers.AufgabenController import _macro_entries
from imswitch.imcontrol.model import aufgaben


def test_bundled_macro_templates():
    templates = {m["name"]: m for m in _macro_entries(aufgaben.EXAMPLES_DIR)}
    assert {"autofocus_here_macro", "shutter_close_macro", "shutter_turret_macro"} <= set(templates)
    assert templates["autofocus_here_macro"]["title"] == "Autofocus here"
    assert set(templates["autofocus_here_macro"]["parameters"]) == {"range_um", "step_um"}
    assert "tile_scan_focus_check" not in templates  # a job, not a macro
