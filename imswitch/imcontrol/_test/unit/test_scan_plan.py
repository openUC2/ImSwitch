"""The request->plan step is pure; these pin the decisions the acquisition loop relies on."""
from imswitch.imcontrol.controller.controllers.experiment_controller.models import ParameterValue, SyntheticChannel
from imswitch.imcontrol.controller.controllers.experiment_controller.scan_plan import (
    resolve_channels, z_offsets, writer_flags)


_REQUIRED = dict(timeLapsePeriod=0, numberOfImages=1, autoFocus=False, autoFocusMin=0, autoFocusMax=0,
                 autoFocusStepSize=1, zStack=False, zStackMin=0, zStackMax=0)


def _p(**kw):
    return ParameterValue(**{**_REQUIRED, **kw})


def test_synthetic_channels_append_in_lockstep():
    plan = resolve_channels(_p(
        illumination=["LED", "Laser"], illuIntensities=[100, 0], exposureTimes=[10, 20], gains=[1, 2],
        syntheticChannels=[SyntheticChannel(name="ring", kind="ring", enabled=True, intensityR=200, radius=3),
                           SyntheticChannel(name="dpc", kind="dpc", enabled=False, intensityG=50)],
        performanceMode=True))
    assert plan.sources == ["LED", "Laser", "ring"]
    assert plan.kinds == ["default", "default", "ring"]
    assert plan.intensities == [100, 0, 200]
    assert len(plan.gains) == len(plan.exposures) == 3
    assert plan.params["ring"]["radius"] == 3 and plan.params["ring"]["intensityR"] == 200
    assert plan.performance_mode is False          # LED-matrix channel forces normal mode
    assert plan.keep_illumination_on is False      # auto + 2 active channels
    assert not plan.passthrough


def test_passthrough_collapses_to_one_inert_sentinel():
    plan = resolve_channels(_p(illumination=["LED"], illuIntensities=[0]))
    assert plan.passthrough
    assert plan.sources == ["LED"] and plan.gains == [-1] and plan.exposures == [0]
    assert plan.keep_illumination_on is True
    assert resolve_channels(_p()).sources == ["default"]


def test_keep_on_setting_overrides_auto():
    assert resolve_channels(_p(illumination=["a"], illuIntensities=[5], keepIlluminationOn="off")).keep_illumination_on is False
    assert resolve_channels(_p(illumination=["a"], illuIntensities=[5])).keep_illumination_on is True


def test_z_offsets_are_relative_and_sorted():
    assert z_offsets(_p(zStack=False, zStackMin=-10, zStackMax=10, zStackStepSize=5)) == [0.0]
    assert z_offsets(_p(zStack=True, zStackMin=-10, zStackMax=10, zStackStepSize=10)) == [-10.0, 0.0, 10.0]
    assert z_offsets(_p(zStack=True, zStackMode="individual", zStackOffsets=[4, -2, 0])) == [-2.0, 0.0, 4.0]


def test_single_position_scan_never_stitches():
    p = _p(ome_write_stitched_tiff=True, ome_write_zarr=True)
    single = writer_flags(p, [[{}], [{}]])
    grid = writer_flags(p, [[{}, {}], [{}]])
    assert single["write_single_tiff"] and not single["write_stitched_tiff"]
    assert grid["write_stitched_tiff"] and not grid["write_single_tiff"]


def test_performance_mode_reads_the_plan_not_phantom_fields():
    # zStackEnabled/zStackStart/illuExposures never existed on ParameterValue: perf
    # mode scanned one plane at a 50 ms default until it was pointed at the plan.
    from imswitch.imcontrol.controller.controllers.experiment_controller.experiment_performance_mode import (
        ExperimentPerformanceMode)
    perf = object.__new__(ExperimentPerformanceMode)
    assert perf._extract_z_stack_parameters({"z_positions": [-10.0, 0.0, 10.0]}) == {"zstart": -10.0, "zstep": 10.0, "nz": 3}
    assert perf._extract_z_stack_parameters({}) == {"zstart": 0.0, "zstep": 0.0, "nz": 1}
    assert perf._exposures_ms({"exposures": [20, 0, None, 35.5]}) == [20.0, 35.5]
