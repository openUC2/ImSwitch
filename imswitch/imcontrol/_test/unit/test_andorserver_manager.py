"""AndorServerManager against andor-server's mock camera.

Needs the andor-server package (https://github.com/beniroquai/andorArm64, directory
andor_server/) on the path, e.g. PYTHONPATH=/path/to/andorArm64; skipped otherwise.
"""

import socket
import threading
import time

import numpy as np
import pytest

andor_server = pytest.importorskip("andor_server")
uvicorn = pytest.importorskip("uvicorn")

from andor_server.api import create_app  # noqa: E402
from andor_server.camera import MockCamera  # noqa: E402
from andor_server.config import Config  # noqa: E402

from imswitch.imcontrol.model.SetupInfo import DetectorInfo  # noqa: E402
from imswitch.imcontrol.model.managers.detectors.AndorServerManager import AndorServerManager  # noqa: E402
from imswitch.imcontrol.model.managers.detectors.DetectorManager import getDetectorExposureSeconds  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = _free_port()
    config = Config(mock=True, port=port, save_dir=str(tmp_path_factory.mktemp("snaps")))
    srv = uvicorn.Server(uvicorn.Config(create_app(config, camera=MockCamera(seed=2)), host="127.0.0.1",
                                        port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield port
    srv.should_exit = True
    thread.join(10)


def make_manager(port, **props):
    info = DetectorInfo(analogChannel=None, digitalLine=None, managerName="AndorServerManager",
                        managerProperties={"host": "127.0.0.1", "port": port, "connectTimeout": 10, **props},
                        forAcquisition=True)
    return AndorServerManager(info, "Andor")


def wait_frame(manager, after, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame, number = manager.getLatestFrame(returnFrameNumber=True)
        if frame is not None and number > after:
            return frame, number
        time.sleep(0.02)
    raise AssertionError("no fresh frame")


@pytest.fixture()
def manager(server):
    m = make_manager(server, andor={"exposure_ms": 20, "live_max_fps": 20, "binning": 1, "roi": None,
                                    "trigger_mode": "internal"}, longExposureMs=400)
    yield m
    m.stopAcquisition()
    m.finalize()


def test_parameters_follow_the_camera(manager):
    params = manager.parameters
    assert manager.model == "MOCK_DU888" and manager.fullShape == (1024, 1024)
    assert params["readout_rate"].options == ["30 MHz", "20 MHz", "10 MHz", "1 MHz"]
    assert params["preamp_gain"].options == ["1x", "2x"]
    assert "blacklevel" not in params  # the mock, like the iXon Ultra 888, has no baseline offset
    assert getDetectorExposureSeconds(manager) == pytest.approx(0.02)


def test_set_parameters_round_trip(manager):
    assert manager.setParameter("exposure", 35) == 35
    assert manager.getParameter("exposure") == 35
    assert manager.setParameter("gain", 5000) == 300  # clamped to the EM gain range
    assert manager.setParameter("output_amplifier", "Conventional") == "Conventional"
    assert manager.parameters["readout_rate"].options == ["1 MHz", "0.1 MHz"]
    assert manager.setParameter("readout_rate", "0.1 MHz") == "0.1 MHz"
    assert manager.setParameter("baseline_clamp", False) is False
    manager.setParameter("output_amplifier", "Electron Multiplying")
    assert manager.setParameter("target_temperature", -60) == -60
    assert manager.setParameter("cooler", True) is True


def test_live_frames_binning_and_crop(manager):
    manager.startAcquisition()
    frame, n = wait_frame(manager, 0)
    assert frame.shape == (1024, 1024) and frame.dtype == np.uint16
    wait_frame(manager, n)
    manager.setBinning(4)
    assert manager.fullShape == (256, 256)
    frame, n = wait_frame(manager, manager.getFrameNumber())
    assert frame.shape == (256, 256)
    manager.crop(10, 20, 50, 40)
    assert manager.shape == (50, 40)
    frame, _ = wait_frame(manager, manager.getFrameNumber())
    assert frame.shape == (40, 50)
    assert manager._camera.settings["roi"] == {"x": 40, "y": 80, "width": 200, "height": 160}
    manager.crop(0, 0, 256, 256)
    assert manager._camera.settings["roi"] is None
    manager.setBinning(1)


def test_long_exposure_runs_as_server_snap(manager):
    manager.setParameter("exposure", 600)
    assert manager.isLongExposure()
    baseline = manager.getFrameNumber()
    manager.startAcquisition()
    time.sleep(0.2)
    assert manager._camera._json("GET", "/status")["state"] == "exposing"
    frame, _ = wait_frame(manager, baseline, timeout=5)
    manager.stopAcquisition()  # within the 0.5 s pause: no second exposure starts
    time.sleep(0.6)
    status = manager._camera._json("GET", "/status")
    assert status["state"] == "idle"
    snaps = manager._camera._json("GET", "/snaps")
    assert snaps[0]["state"] == "done" and snaps[0]["saved_path"]


def test_software_trigger_snap_sync(manager):
    assert manager.setTriggerSource("software") is True
    frame = manager.snapSync(timeout=5)
    assert frame is not None and frame.shape == (1024, 1024)
    manager.setTriggerSource("Continous")
    assert manager.getCurrentTriggerType() == "Continous"


def test_server_unreachable_comes_up_disconnected():
    m = make_manager(_free_port(), connectTimeout=0.5)
    try:
        assert not m._camera.is_connected
        assert m.getLatestFrame() is None
        assert m.fullShape == (1024, 1024)
    finally:
        m.finalize()
