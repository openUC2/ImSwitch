"""The autofocus a run uses is the one the Z / Focus panel shows."""
from imswitch.imcontrol.controller.controllers.experiment_controller.models import ParameterValue

BASE = dict(timeLapsePeriod=0, numberOfImages=1, autoFocus=False, zStack=False,
            zStackMin=0, zStackMax=0)


def test_the_panel_range_and_step_reach_the_run():
    p = ParameterValue(**BASE, autoFocusMin=0, autoFocusMax=0, autoFocusStepSize=0.1,
                       autoFocusRange=150, autoFocusResolution=15)
    kwargs = p.autofocus_kwargs()
    assert kwargs["af_range"] == 150 and kwargs["af_resolution"] == 15


def test_clients_without_the_panel_fields_keep_min_max_step():
    p = ParameterValue(**BASE, autoFocusMin=-50, autoFocusMax=50, autoFocusStepSize=5)
    assert p.autofocus_kwargs()["af_range"] == 50
    assert p.autofocus_kwargs()["af_resolution"] == 5


def test_a_sweep_never_takes_thousands_of_planes():
    # what one browser had saved: range 0 -> 100 µm fallback, step 0.1 µm (2000 planes)
    p = ParameterValue(**BASE, autoFocusMin=0, autoFocusMax=0, autoFocusStepSize=0.1)
    kwargs = p.autofocus_kwargs()
    assert kwargs["af_range"] == 100
    planes = 2 * kwargs["af_range"] / kwargs["af_resolution"]
    assert planes <= ParameterValue.MAX_AUTOFOCUS_PLANES - 1


class _Lights:
    """Run state AutofocusMixin._light_for_autofocus reads, and what it switched."""

    def __init__(self, sources, intensities, passthrough=False, device_value=0):
        quiet = lambda *a, **k: None  # noqa: E731
        self._logger = type("L", (), {"debug": quiet, "warning": quiet})()
        self._illuminationSources = sources
        self._illuminationIntensities = intensities
        self._illuminationPassthrough = passthrough
        self.allIlluNames = ["LED", "Laser488"]
        laser = type("Laser", (), {"getValue": lambda _self: device_value})()
        self._master = type("M", (), {"lasersManager": {n: laser for n in self.allIlluNames}})()
        self.switched = []

    def set_laser_power(self, power, channel):
        self.switched.append((channel, power))

    def light(self, channel=""):
        from imswitch.imcontrol.controller.controllers.experiment_controller.autofocus_api import (
            AutofocusMixin)
        return AutofocusMixin._light_for_autofocus(self, channel)


def test_autofocus_without_a_channel_looks_through_the_runs_first_light_that_is_on():
    # the focus-map phase runs before any tile lights the sample: it focused in the dark
    run = _Lights(["LED", "Laser488"], [0, 700])
    assert run.light() == "Laser488" and run.switched == [("Laser488", 700)]


def test_autofocus_with_its_own_channel_uses_the_runs_intensity_for_it():
    run = _Lights(["Laser488", "LED"], [300, 1023])
    assert run.light("LED") == "LED" and run.switched == [("LED", 1023)]
    other = _Lights([], [], device_value=0)
    assert other.light("LED") == "LED" and other.switched == [("LED", 50)]   # safe default


def test_autofocus_leaves_a_light_set_by_hand_alone():
    # passthrough: no light in the experiment, power 1 is the plan's sentinel
    run = _Lights(["LED"], [1], passthrough=True)
    assert run.light() == "" and run.switched == []
    assert _Lights(["LED"], [0]).light() == ""


class _FocusPoint:
    """A focus-map point: autofocus results in turn, and what the last curve looked like."""

    def __init__(self, results, quality):
        quiet = lambda *a, **k: None  # noqa: E731
        self._logger = type("L", (), {"info": quiet, "warning": quiet, "debug": quiet})()
        self.results = list(results)
        self.moves = []
        af = type("AF", (), {"_lastFocusQuality": quality})()
        self._master = type("M", (), {"getController": lambda _self, name: af})()

    def autofocus(self, **kwargs):
        return self.results.pop(0)

    def move_stage_z(self, z, relative=False):
        self.moves.append(z)

    def run(self):
        from imswitch.imcontrol.controller.controllers.experiment_controller.focus_map_api import (
            FocusMapMixin)
        self.FOCUS_EDGE_RETRIES = FocusMapMixin.FOCUS_EDGE_RETRIES
        return FocusMapMixin._autofocus_following_edge(self, {}, "tissue_Slide2_1")


def test_a_focus_beyond_the_window_is_searched_again_around_its_edge():
    # slide 2 lay below the window centred on slide 3's points: every point failed
    edge = {"peak_at_edge": True, "z_argmax": 7290.9, "contrast": 0.8}
    point = _FocusPoint([None, 7262.0], edge)
    assert point.run() == 7262.0 and point.moves == [7290.9]


def test_only_one_window_more_and_never_on_a_flat_curve():
    edge = {"peak_at_edge": True, "z_argmax": 7290.9, "contrast": 0.8}
    point = _FocusPoint([None, None, 7100.0], edge)
    assert point.run() is None and point.moves == [7290.9]     # at most 2x the range
    flat = _FocusPoint([None, 7262.0], {"peak_at_edge": True, "z_argmax": 7290.9,
                                        "contrast": 0.05})
    assert flat.run() is None and flat.moves == []              # no texture: nothing to follow
    found = _FocusPoint([7300.0], edge)
    assert found.run() == 7300.0 and found.moves == []
