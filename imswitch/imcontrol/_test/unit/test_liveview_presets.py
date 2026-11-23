"""Live-view presets are stored per instrument, written atomically."""
import types

from imswitch.imcontrol.controller.controllers.LiveViewController import LiveViewController


def _call(name, ctl, *args):
    fn = getattr(LiveViewController, name)
    return getattr(fn, "__wrapped__", fn)(ctl, *args)


def test_presets_round_trip(tmp_path):
    ctl = types.SimpleNamespace(_presetsPath=lambda: str(tmp_path / "config" / "liveview_presets.json"))
    assert _call("getLiveViewPresets", ctl) == {"presets": None}  # never saved
    presets = [{"id": "a", "name": "brightfield", "streamSettings": {"subsampling": 2}}]
    assert _call("setLiveViewPresets", ctl, {"presets": presets})["count"] == 1
    assert _call("getLiveViewPresets", ctl) == {"presets": presets}
    assert _call("setLiveViewPresets", ctl, {"presets": "nope"})["status"] == "error"
    assert _call("getLiveViewPresets", ctl) == {"presets": presets}  # bad input kept the list
