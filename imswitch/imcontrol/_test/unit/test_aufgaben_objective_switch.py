"""A job that changes the objective mid-run: images after the switch carry the new pixel size."""
import pytest
import zarr

from imswitch.imcontrol.model import aufgaben as A
from imswitch.imcontrol.model.aufgaben.engine import Run
from imswitch.imcontrol.model.aufgaben.simulated import SimulatedMicroscope
from imswitch.imcontrol.model.aufgaben.storage import OmeStore

pytest.importorskip("ome_zarr_models")
PIXEL_UM = {0: 1.0, 1: 0.25}   # per turret slot


class Turret(SimulatedMicroscope):
    """The camera pixel size follows the objective, as ObjectiveController sets it."""

    def set_objective(self, slot):
        super().set_objective(slot)
        self.px = PIXEL_UM[slot]


def test_images_after_an_objective_switch_carry_its_pixel_size(tmp_path):
    job = A.Job.model_validate({
        "name": "t",
        "resources": {"bf": {"kind": "capture", "channels": [{"config": "LED", "exposure": 5}]}},
        "tasks": [{"id": "low", "task": "capture", "params": {"channels": "@bf"}},
                  {"id": "switch", "task": "set_objective", "params": {"slot": 1}},
                  {"id": "high", "task": "capture", "params": {"channels": "@bf"}}]})
    run = Run(job, Turret(pixel_size_um=PIXEL_UM[0]), sink=OmeStore(tmp_path))
    assert run.run(confirmed=True) == "finished", run.message
    for node, px in (("low", PIXEL_UM[0]), ("high", PIXEL_UM[1])):
        attrs = zarr.open_group(str(tmp_path / f"{node}.ome.zarr"), mode="r").attrs
        scale = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale[-2:] == [px, px], node          # the file's x/y pixel size
        assert run.seen[node].pixel_size_um == px, node   # what analysis steps measure with
