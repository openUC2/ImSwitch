"""RecordingService streaming: frames in, one valid NGFF image per detector out.
Snaps: OME-TIFF or OME-Zarr through the same writer, and a snap that wrote nothing says why."""
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from ome_zarr_models.v04.image import Image

from imswitch.imcontrol.model.io.recording_service import RecordingService, SaveFormat


class _Camera:
    shape = (40, 30)               # width, height in px
    pixelSizeUm = [1, 1, 0.65]
    dtype = "uint16"


class _Detectors:
    def getAllDeviceNames(self):
        return ["cam"]

    def __getitem__(self, name):
        return _Camera()


def test_streaming_writes_valid_ngff(tmp_path):
    service = RecordingService()
    try:
        assert service.start_streaming(str(tmp_path / "rec"), _Detectors(), auto_capture=False)
        assert service.is_streaming and service.is_recording
        assert not service.start_streaming(str(tmp_path / "again"), _Detectors())
        for t in range(3):
            service.write_streaming_frame("cam", np.full((30, 40), t, np.uint16))
        stats = service.stop_streaming()
    finally:
        service.shutdown()
    assert stats["frames"] == 3 and not service.is_streaming
    group_path = next(p.parent for p in Path(tmp_path).rglob(".zattrs")
                      if (p.parent / "0").exists())
    group = zarr.open_group(str(group_path), mode="r")
    Image.from_zarr(group)
    assert group["0"].shape == (3, 1, 1, 30, 40) and int(group["0"][2, 0, 0, 5, 5]) == 2


def test_snap_tiff_keeps_calibration_position_and_attributes(tmp_path):
    """Snaps are OME-TIFFs: pixel size, top-left position, exposure, all attributes kept."""
    import ome_types
    from imswitch.imcontrol.model.io.recording_service import save_snap
    metadata = {"Detector": "cam", "Channel": "488nm", "PhysicalSizeX": 0.5,
                "PositionX": 100.0, "PositionY": 50.0, "PositionZ": 3.0, "ExposureTime": 0.02,
                "Gain": 2.0, "AcquisitionAttributes": {"Laser": {"power": 5}}}
    path = tmp_path / "snap.ome.tiff"
    save_snap(str(path), np.zeros((30, 40), np.uint16), metadata)
    ome = ome_types.from_tiff(path)
    pixels = ome.images[0].pixels
    plane = pixels.planes[0]
    assert pixels.physical_size_x == 0.5 and pixels.channels[0].name == "488nm"
    assert (plane.position_x, plane.position_y, plane.exposure_time) == (90.0, 42.5, 20.0)
    notes = {m.k: m.value for m in ome.structured_annotations.map_annotations[0].value.ms}
    assert notes["Gain"] == "2.0" and notes["AcquisitionAttributes/Laser/power"] == "5"


@pytest.mark.parametrize("zarr_too", [False, True])
def test_streaming_tiff_is_exported_from_the_zarr_image(tmp_path, zarr_too):
    """OME-TIFF recordings come from the same NGFF image; TIFF-only keeps no zarr."""
    import ome_types
    service = RecordingService()
    try:
        service.start_streaming(str(tmp_path / "rec"), _Detectors(), write_zarr=zarr_too,
                                write_tiff=True, auto_capture=False)
        for t in range(4):
            service.write_streaming_frame("cam", np.full((30, 40), t, np.uint16))
        service.stop_streaming()
    finally:
        service.shutdown()
    tiff = next(Path(tmp_path).rglob("*_cam.ome.tiff"))
    pixels = ome_types.from_tiff(tiff).images[0].pixels
    assert (pixels.size_t, pixels.size_y, pixels.size_x) == (4, 30, 40)
    assert pixels.physical_size_x == 0.65
    stores = [p for p in Path(tmp_path).rglob("*.zarr") if p.is_dir()]
    assert bool(stores) == zarr_too


def _cameras(frame):
    """A detectorsManager stand-in: one camera, "cam", whose latest frame is ``frame``."""
    return {"cam": SimpleNamespace(getLatestFrame=lambda: frame)}


@pytest.mark.parametrize("async_write", [False, True])
def test_zarr_snap_is_ome_zarr_with_pixel_size_and_objective(tmp_path, monkeypatch, async_write):
    from imswitch.imcontrol.model.aufgaben import storage
    objective = {"name": "Plan 20x", "na": 0.75, "magnification": 20, "immersion": "Air"}
    monkeypatch.setattr(storage, "current_objective", lambda: objective)
    frame = np.arange(30 * 40, dtype=np.uint16).reshape(30, 40)
    service, told = RecordingService(_cameras(frame)), []
    try:
        result = service.snap(["cam"], str(tmp_path / "snap"), format=SaveFormat.OME_ZARR,
                              attrs={"cam": {"PixelSizeUm": 0.5}}, async_write=async_write,
                              callback=told.append)["cam"]
        service._storage_worker._task_queue.join()
    finally:
        service.shutdown()
    assert result.success and result.filepath == str(tmp_path / "snap_cam.ome.zarr")
    assert [r.success for r in told] == ([True] if async_write else [])
    group = zarr.open_group(result.filepath, mode="r")
    Image.from_zarr(group)
    scale = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale[-2:] == [0.5, 0.5] and np.array_equal(group["0"][0, 0, 0], frame)
    acquisition = group.attrs["acquisition"]
    assert {k: acquisition[f"objective_{k}"] for k in objective} == objective


def _snap_controller(tmp_path, monkeypatch, frame):
    """RecordingController with what startSnap needs, over the real RecordingService."""
    from imswitch.imcommon.model import dirtools
    from imswitch.imcontrol.controller.controllers.RecordingController import RecordingController
    monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath", lambda: str(tmp_path))
    quiet = lambda *args, **kwargs: None   # noqa: E731
    ctrl = object.__new__(RecordingController)
    ctrl._RecordingController__logger = SimpleNamespace(debug=quiet, info=quiet, warning=quiet,
                                                        error=quiet)
    ctrl._master = SimpleNamespace(detectorsManager=_cameras(frame))
    ctrl._snapJobs, ctrl._snapJobsLock, ctrl._lastSnapJobId = {}, threading.Lock(), None
    ctrl._recording_service = RecordingService(ctrl._master.detectorsManager)
    ctrl.updateRecAttrs = ctrl._refuseStandInCamera = ctrl._disarmDetectorsAfterSnap = quiet
    ctrl._armDetectorsForSnap = lambda names, cancelEvent=None: []
    ctrl.getDetectorNamesToCapture = lambda: ["cam"]
    ctrl._get_detector_attrs = lambda name: {}
    return ctrl


@pytest.mark.parametrize("fmt, status, outcome", [
    (SaveFormat.OME_ZARR, "done", "_x_cam.ome.zarr"), (SaveFormat.ZARR, "done", "_x_cam.ome.zarr"),
    (SaveFormat.OME_TIFF, "done", "_x_cam.ome.tiff"),
    (SaveFormat.MP4, "error", "Snap failed: cam: Unsupported format for snap: SaveFormat.MP4")])
def test_snap_job_is_done_only_when_a_file_was_written(tmp_path, monkeypatch, fmt, status,
                                                       outcome):
    """startSnap + getSnapStatus: "done" with the file, or "error" with the reason."""
    ctrl = _snap_controller(tmp_path, monkeypatch, np.ones((30, 40), np.uint16))
    try:
        job_id = ctrl.startSnap(fileName="x", saveFormat=fmt.value)["jobId"]
        deadline = time.time() + 10
        while (state := ctrl.getSnapStatus(job_id))["status"] in ("pending", "running"):
            assert time.time() < deadline, state
            time.sleep(0.01)
    finally:
        ctrl._recording_service.shutdown()
    assert state["status"] == status, state
    written = list(tmp_path.rglob("*_x_cam.*"))
    if status == "done":
        assert "error" not in state and state["result"]["fullPath"].endswith(outcome)
        assert [str(p) for p in written] == [state["result"]["fullPath"]]
    else:
        assert state["error"].startswith(outcome) and not written and "result" not in state
