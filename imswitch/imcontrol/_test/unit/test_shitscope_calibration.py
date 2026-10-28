"""StageCalibration against a simulated PCB stage: static friction threshold,
reversal loss, optional stuck axis / yaw, and a camera that images a textured
world at the sled's true position."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, rotate

from imswitch.imcontrol.model.shitscope_calibration import StageCalibration, rotation_deg

RNG = np.random.default_rng(11)
WORLD = 1000 + 400 * gaussian_filter(RNG.normal(size=(3000, 3000)), 3)
UM_PX, H, W = 2.0, 600, 800


class SimStage:
    def __init__(self, threshold=(5000, 12000), um_per_step=(130.0, 125.0), loss_um=230.0,
                 yaw_after_moves=None, yaw_deg=2.0, world=WORLD):
        self.thr = dict(zip("XY", threshold)); self.ups = dict(zip("XY", um_per_step))
        self.loss, self.world = loss_um, world
        self.power = {"X": 0, "Y": 0}; self.pos = {"X": 0.0, "Y": 0.0}; self.last_dir = {"X": 0, "Y": 0}
        self.cmd = [0, 0]; self.moves = 0
        self.yaw_after, self.yaw_deg, self.yaw = yaw_after_moves, yaw_deg, 0.0

    def set_power(self, px, py):
        self.power = {"X": px, "Y": py}

    def move_steps(self, dx, dy):
        self.moves += 1; self.cmd[0] += dx; self.cmd[1] += dy
        for axis, n in (("X", dx), ("Y", dy)):
            if not n or self.power[axis] < self.thr[axis]:
                continue                                        # static friction wins
            d = int(np.sign(n))
            travel = n * self.ups[axis] - (d * self.loss if self.last_dir[axis] not in (0, d) else 0)
            if d * travel < 0:
                travel = 0.0
            self.pos[axis] += travel + RNG.normal(0, 3); self.last_dir[axis] = d
        if self.yaw_after is not None and self.moves >= self.yaw_after:
            self.yaw = self.yaw_deg

    def grab(self):
        return self._render() + RNG.normal(0, 5, (H, W))           # fresh camera noise per frame

    def _render(self):
        x = int(round(self.pos["X"] / UM_PX)) + 1100; y = int(round(self.pos["Y"] / UM_PX)) + 1100
        if not self.yaw:
            return self.world[y:y + H, x:x + W]
        big = self.world[y - 100:y + H + 100, x - 100:x + W + 100]
        return rotate(big, self.yaw, reshape=False, order=1)[100:100 + H, 100:100 + W]


def calibrate(sim, **kw):
    cal = StageCalibration(sim.grab, sim.move_steps, sim.set_power, UM_PX, start_power=(18000, 24000), **kw)
    return cal, cal.run()


def test_rotation_estimate():
    a = WORLD[1000:1400, 1000:1400]
    b = rotate(WORLD[900:1500, 900:1500], 2.0, reshape=False, order=1)[100:500, 100:500]
    assert abs(abs(rotation_deg(a, b)) - 2.0) < 0.3 and abs(rotation_deg(a, a)) < 0.1


def test_finds_threshold_step_size_and_reversal_loss():
    sim = SimStage()
    cal, r = calibrate(sim)
    assert cal.state == "done" and r["ok"], (r["warnings"], cal.log)
    x, y = r["axes"]["X"], r["axes"]["Y"]
    assert 5000 <= x["threshold_power"] < 5000 / 0.8 and 12000 <= y["threshold_power"] < 12000 / 0.8
    assert r["recommended"]["powerX"] == pytest.approx(1.5 * x["threshold_power"], abs=500)
    assert abs(x["um_per_step"] - 130) < 6 and abs(y["um_per_step"] - 125) < 6
    assert abs(x["reversal_loss_um"] - 230) < 60 and r["recommended"]["approachOvershootSteps"] == 4
    assert abs(abs(x["image_angle_deg"] - y["image_angle_deg"]) - 90) < 3
    assert sim.cmd == [0, 0] and sim.power == {"X": 18000, "Y": 24000}   # stage and drive restored


def test_stuck_axis_is_reported():
    sim = SimStage(threshold=(5000, 40000))
    cal, r = calibrate(sim)
    assert not r["ok"] and "no_motion" in r["flags"] and r["axes"]["Y"] == {"moves": False}
    assert "powerY" not in r["recommended"] and "Y does not move" in r["warnings"][0]
    assert sim.cmd == [0, 0]


def test_sled_yaw_is_reported():
    sim = SimStage(yaw_after_moves=30, yaw_deg=2.5)
    cal, r = calibrate(sim)
    assert "rotated" in r["flags"] and abs(abs(r["rotation_deg"]) - 2.5) < 0.5


def test_blank_sample_aborts():
    flat = np.full_like(WORLD, 1000.0)                              # only camera noise
    sim = SimStage(world=flat)
    cal, r = calibrate(sim)
    assert cal.state == "error" and "texture" in cal.error and sim.moves == 0


def test_stop():
    sim = SimStage()
    cal = StageCalibration(sim.grab, sim.move_steps, sim.set_power, UM_PX)
    cal.stop(); cal.run()
    assert cal.state == "stopped" and sim.cmd == [0, 0]
