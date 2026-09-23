import datetime
import json
import os
import time
from threading import Thread, Event
from typing import Any, Dict

import cv2
import numpy as np
import tifffile as tif

from imswitch.imcommon.model import dirtools, APIExport
from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import initLogger
from ..basecontrollers import LiveUpdatedController


# EcoTaxa-flavoured sample metadata, the subset PlanktoScope/FairScope writes into
# metadata.json next to the acquired frames. Empty strings/zeros mean "not filled in".
DEFAULT_METADATA = {
    "sample_project": "",
    "sample_id": "",
    "sample_ship": "",
    "sample_operator": "",
    "sample_gear": "",              # net / sampling gear, e.g. "Bongo net"
    "sample_mesh_size_um": 0,
    "sample_depth_min_m": 0.0,
    "sample_depth_max_m": 0.0,
    "sample_latitude": 0.0,         # decimal degrees, +N
    "sample_longitude": 0.0,        # decimal degrees, +E
    "sample_date": "",              # ISO yyyy-mm-dd
    "sample_time": "",              # HH:MM
    "sample_total_volume_ml": 0.0,
    "acq_instrument": "openUC2 FlowStop",
    "acq_celltype_ul": 0.0,         # flow cell volume
    "object_notes": "",
}

# Acquisition parameters, stored under these keys in the FlowStopManager config.json.
PARAM_KEYS = ("experimentName", "experimentDescription", "uniqueId", "numImages",
              "volumePerImage", "timeToStabilize", "pumpSpeed", "frameRate",
              "fileFormat", "isRecordVideo", "delayTimeAfterRestart", "pumpTimeout",
              "wasRunning")


class FlowStopController(LiveUpdatedController):
    """Flow-cell imaging: step the pump by a fixed volume, settle, snap, repeat.

    Distances/volumes are in motor steps at this layer (the pump axis is a stepper);
    conversion to ml is the user's calibration and lives in the metadata.
    """

    sigImageReceived = Signal()
    sigImagesTaken = Signal(int)
    sigIsRunning = Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        cfg = self._master.FlowStopManager.defaultConfig
        self.pumpAxis = cfg.get("axisFlow", "X")
        self.focusAxis = cfg.get("axisFocus", "Z")
        self.defaultSavePath = cfg.get("savePath", "./")
        self.mExperimentParameters = self._paramsFromConfig(cfg)
        self.mMetadata = {**DEFAULT_METADATA, **cfg.get("metadata", {})}

        self.tSettle = 0.05
        self.imagesTaken = 0
        self.is_measure = False
        self.isRecordVideo = False
        self.video_safe = None
        self.thread = None
        self._stopEvent = Event()
        self._startTime = 0.0
        self._endTime = 0.0
        self._relativePath = ""
        self._lastError = ""

        # select detector / illumination / stage; a missing device must not kill startup
        self.detectorFlowCam = self._firstDevice(self._master.detectorsManager)
        self.ledSource = self._firstDevice(self._master.lasersManager)
        if self.ledSource is not None:
            self.ledSource.setEnabled(1)
        self.positionerName = None
        self.positioner = None
        positionerNames = self._master.positionersManager.getAllDeviceNames()
        if positionerNames:
            self.positionerName = positionerNames[0]
            self.positioner = self._master.positionersManager[self.positionerName]

        self.changeAutoExposureTime('auto')

        # resume an experiment that was running when the software was restarted
        if cfg.get("wasRunning", False):
            p = dict(self.mExperimentParameters)
            p["delayToStart"] = cfg.get("delayTimeAfterRestart", 1)
            self._startExperiment(p)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _firstDevice(manager):
        names = manager.getAllDeviceNames()
        return manager[names[0]] if names else None

    @staticmethod
    def _paramsFromConfig(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "experimentName": cfg.get("experimentName", "FlowStopExperiment"),
            "experimentDescription": cfg.get("experimentDescription", ""),
            "uniqueId": str(cfg.get("uniqueId", "")),
            "numImages": int(cfg.get("numImages", -1)),
            "volumePerImage": float(cfg.get("volumePerImage", 1000)),
            "timeToStabilize": float(cfg.get("timeToStabilize", 0.5)),
            "pumpSpeed": float(cfg.get("pumpSpeed", 10000)),
            "frameRate": float(cfg.get("frameRate", 1)),
            "fileFormat": cfg.get("fileFormat", "JPG"),
            "isRecordVideo": bool(cfg.get("isRecordVideo", False)),
            "delayTimeAfterRestart": float(cfg.get("delayTimeAfterRestart", 1)),
            "pumpTimeout": float(cfg.get("pumpTimeout", 5.0)),
            "wasRunning": bool(cfg.get("wasRunning", False)),
        }

    def _persist(self):
        """Write the current parameters + metadata back to the FlowStopManager config."""
        try:
            cfg = self._master.FlowStopManager.defaultConfig
            cfg.update({k: self.mExperimentParameters[k]
                        for k in PARAM_KEYS if k in self.mExperimentParameters})
            cfg["metadata"] = self.mMetadata
            self._master.FlowStopManager.writeConfig(cfg)
        except Exception as e:
            self._logger.error(f"Could not persist FlowStop config: {e}")

    # --------------------------------------------------------------- parameters

    @APIExport()
    def getExperimentParameters(self) -> dict:
        self.mExperimentParameters["timeStamp"] = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
        return self.mExperimentParameters

    @APIExport()
    def isRunning(self) -> bool:
        return self.is_measure

    @APIExport(requestType="POST")
    def setFlowStopParameters(self, params: Dict[str, Any]) -> dict:
        """Update and persist the acquisition parameters. Unknown keys are ignored."""
        for k in PARAM_KEYS:
            if k in params:
                self.mExperimentParameters[k] = params[k]
        self.mExperimentParameters = self._paramsFromConfig(
            {**self._master.FlowStopManager.defaultConfig, **self.mExperimentParameters})
        self._persist()
        return self.mExperimentParameters

    @APIExport()
    def getFlowStopMetadata(self) -> dict:
        """EcoTaxa-style sample metadata written next to each acquisition."""
        return self.mMetadata

    @APIExport(requestType="POST")
    def setFlowStopMetadata(self, metadata: Dict[str, Any]) -> dict:
        """Update and persist the sample metadata. Only known keys are kept."""
        self.mMetadata = {**self.mMetadata,
                          **{k: v for k, v in metadata.items() if k in DEFAULT_METADATA}}
        self._persist()
        return self.mMetadata

    @APIExport()
    def getFlowStopStatus(self) -> dict:
        """Progress of the running (or last) acquisition.

        `relativePath` is relative to the data folder, so it can be listed/thumbnailed
        through the FileManager endpoints. `progress` is -1 for an unlimited run.
        """
        numImages = int(self.mExperimentParameters.get("numImages", -1))
        if not self._startTime:
            elapsed = 0.0
        else:
            # freeze the clock once the run is over, so an idle status stops ticking
            end = time.time() if self.is_measure else (self._endTime or time.time())
            elapsed = end - self._startTime
        progress, eta = -1.0, -1.0
        if numImages > 0:
            progress = min(1.0, self.imagesTaken / numImages)
            if self.imagesTaken > 0 and self.is_measure:
                eta = (elapsed / self.imagesTaken) * (numImages - self.imagesTaken)
        return {
            "isRunning": self.is_measure,
            "imagesTaken": self.imagesTaken,
            "numImages": numImages,
            "progress": progress,
            "elapsedSeconds": round(elapsed, 1),
            "etaSeconds": round(eta, 1),
            "experimentName": self.mExperimentParameters.get("experimentName", ""),
            "relativePath": self._relativePath,
            "diskUsage": dirtools.getDiskusage(),
            "lastError": self._lastError,
        }

    @APIExport()
    def getFlowStopHardware(self) -> dict:
        """Devices and ranges the UI needs to render its controls."""
        illu = {}
        if self.ledSource is not None:
            illu = {
                "name": self.ledSource.name,
                "min": self.ledSource.valueRangeMin,
                "max": self.ledSource.valueRangeMax,
                "step": self.ledSource.valueRangeStep,
                "units": self.ledSource.valueUnits,
            }
        positions = {}
        try:
            if self.positioner is not None:
                positions = self.positioner.getPosition()
        except Exception:
            pass
        return {
            "detectorName": getattr(self.detectorFlowCam, "name", None),
            "detectorNames": self._master.detectorsManager.getAllDeviceNames(),
            "illumination": illu,
            "illuminationNames": self._master.lasersManager.getAllDeviceNames(),
            "positionerName": self.positionerName,
            "pumpAxis": self.pumpAxis,
            "focusAxis": self.focusAxis,
            "positions": positions,
        }

    # ------------------------------------------------------------ manual control

    @APIExport(runOnUIThread=True)
    def stopPump(self):
        if self.positioner is not None:
            self.positioner.stopAll()

    @APIExport(runOnUIThread=True)
    def movePump(self, value: float = 0.0, speed: float = 10000.0):
        """Jog the pump axis by `value` steps (relative, non-blocking)."""
        if self.positioner is not None:
            self.positioner.move(value=value, speed=speed, axis=self.pumpAxis,
                                 is_absolute=False, is_blocking=False)

    @APIExport(runOnUIThread=True)
    def moveFocus(self, value: float = 0.0, speed: float = 10000.0):
        """Jog the focus axis by `value` steps (relative, non-blocking)."""
        if self.positioner is not None:
            self.positioner.move(value=value, speed=speed, axis=self.focusAxis,
                                 is_absolute=False, is_blocking=False)

    @APIExport(runOnUIThread=True)
    def stopFocus(self):
        if self.positioner is not None:
            self.positioner.stopAll()

    @APIExport(runOnUIThread=True)
    def getCurrentFrameNumber(self):
        return self.imagesTaken

    @APIExport(runOnUIThread=True)
    def setIlluIntensity(self, value: float = 0.0, enabled: bool = True):
        if self.ledSource is None:
            return
        self.ledSource.setValue(value)
        self.ledSource.setEnabled(bool(enabled))

    @APIExport(runOnUIThread=True)
    def changeExposureTime(self, value):
        """ Change exposure time. """
        self.detectorFlowCam.setParameter(name="exposure", value=value)

    @APIExport(runOnUIThread=True)
    def changeAutoExposureTime(self, value):
        """ Change auto exposure mode ('auto' / 'manual'). """
        try:
            self.detectorFlowCam.setParameter(name="exposure_mode", value=value)
        except Exception as e:
            self._logger.error(f"Could not set auto exposure mode: {e}")

    # --------------------------------------------------------------- experiment

    @APIExport(runOnUIThread=True)
    def startFlowStopExperimentFastAPI(self, timeStamp: str = "", experimentName: str = "",
                                       experimentDescription: str = "", uniqueId: str = "",
                                       numImages: int = -1, volumePerImage: float = 1000,
                                       timeToStabilize: float = 0.5, delayToStart: float = 0,
                                       frameRate: float = 1, filePath: str = "./",
                                       fileFormat: str = "JPG", isRecordVideo: bool = False,
                                       pumpSpeed: float = 10000):
        """Start an acquisition. Returns the parameters actually used."""
        params = dict(self.mExperimentParameters)
        params.update({
            "experimentName": experimentName or params["experimentName"],
            "experimentDescription": experimentDescription,
            "uniqueId": uniqueId,
            "numImages": int(numImages),
            "volumePerImage": float(volumePerImage),
            "timeToStabilize": float(timeToStabilize),
            "frameRate": float(frameRate),
            "fileFormat": fileFormat,
            "isRecordVideo": bool(isRecordVideo),
            "pumpSpeed": float(pumpSpeed),
            "delayToStart": float(delayToStart),
        })
        if filePath and filePath != "./":
            self.defaultSavePath = filePath
        return self._startExperiment(params)

    @APIExport(requestType="POST")
    def startFlowStopExperiment(self, params: Dict[str, Any] = None) -> dict:
        """Start an acquisition from the stored parameters, optionally overridden."""
        merged = dict(self.mExperimentParameters)
        merged.update(params or {})
        return self._startExperiment(merged)

    def _startExperiment(self, params: Dict[str, Any]) -> dict:
        if self.thread is not None and self.thread.is_alive():
            self._logger.warning("FlowStop experiment already running.")
            return self.getFlowStopStatus()
        try:
            params["uniqueId"] = int(params.get("uniqueId"))
        except (TypeError, ValueError):
            params["uniqueId"] = int(np.random.randint(0, 2 ** 16))
        params["timeStamp"] = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")

        self.mExperimentParameters = {**self.mExperimentParameters, **params}
        self._lastError = ""
        self._stopEvent.clear()
        self.thread = Thread(target=self.flowExperimentThread, name="FlowStopExperiment",
                             args=(dict(self.mExperimentParameters),), daemon=True)
        self.thread.start()
        return self.mExperimentParameters

    @APIExport(runOnUIThread=True)
    def stopFlowStopExperiment(self):
        self._stopEvent.set()
        self.is_measure = False
        self.sigIsRunning.emit(False)
        return self.getFlowStopStatus()

    def flowExperimentThread(self, p: Dict[str, Any]):
        """Pump a fixed volume, settle, snap, repeat until `numImages` or stopped.

        A `numImages` <= 0 means "run until stopped". All sleeps go through the stop
        event so a stop request takes effect within one pump step.
        """
        timeStamp = p["timeStamp"]
        numImages = int(p.get("numImages", -1))
        volumePerImage = float(p.get("volumePerImage", 1000))
        timeToStabilize = float(p.get("timeToStabilize", 0.5))
        pumpSpeed = float(p.get("pumpSpeed", 10000))
        pumpTimeout = float(p.get("pumpTimeout", 5.0))
        frameRate = float(p.get("frameRate", 1))
        fileFormat = p.get("fileFormat", "JPG")
        self.isRecordVideo = bool(p.get("isRecordVideo", False))

        delayToStart = abs(float(p.get("delayToStart", 0)))
        if delayToStart and self._stopEvent.wait(delayToStart):
            return
        self._commChannel.sigStartLiveAcquistion.emit(True)

        drivePath = dirtools.UserFileDirs.getValidatedDataPath()
        self._relativePath = os.path.join('recordings', timeStamp)
        dirPath = os.path.join(drivePath, self._relativePath)
        os.makedirs(dirPath, exist_ok=True)
        self._writeMetadataFile(dirPath, p)

        self.imagesTaken = 0
        self._startTime = time.time()
        self._endTime = 0.0
        self.is_measure = True
        self.sigIsRunning.emit(True)
        self.sigImagesTaken.emit(0)

        if self.isRecordVideo:
            self.video_safe = VideoSafe(self.detectorFlowCam.getLatestFrame,
                                        output_folder=dirPath, frame_rate=5)
            self.video_safe.start()
        try:
            while self.is_measure and not self._stopEvent.is_set():
                if 0 < numImages <= self.imagesTaken:
                    break
                if dirtools.getDiskusage() > .95:
                    self._lastError = "Disk is full - acquisition stopped."
                    self._logger.error(self._lastError)
                    break
                tFrame = time.time()
                self.positioner.move(value=volumePerImage, speed=pumpSpeed, axis=self.pumpAxis,
                                     is_absolute=False, is_blocking=True, timeout=pumpTimeout)
                if self._stopEvent.wait(timeToStabilize):
                    break

                mFileName = (f'{timeStamp}_{p.get("experimentName", "")}_'
                             f'{p.get("uniqueId", "")}_{self.imagesTaken:05d}')
                self.snapImageFlowCam(os.path.join(dirPath, mFileName), fileFormat=fileFormat)
                self.imagesTaken += 1
                self.sigImagesTaken.emit(self.imagesTaken)

                if frameRate > 0:
                    remaining = (1.0 / frameRate) - (time.time() - tFrame)
                    if remaining > 0 and self._stopEvent.wait(remaining):
                        break
        except Exception as e:
            self._lastError = str(e)
            self._logger.error(f"FlowStop experiment failed: {e}")
        finally:
            if self.isRecordVideo and self.video_safe is not None:
                self.video_safe.stop()
                self.video_safe = None
            self.is_measure = False
            self._endTime = time.time()
            self.sigIsRunning.emit(False)
            self._logger.debug(f"FlowStop experiment finished after {self.imagesTaken} images.")

    def _writeMetadataFile(self, dirPath: str, p: Dict[str, Any]):
        """Write the EcoTaxa-style metadata.json next to the frames (PlanktoScope layout)."""
        meta = {
            **self.mMetadata,
            "acq_id": p.get("uniqueId"),
            "acq_name": p.get("experimentName"),
            "acq_description": p.get("experimentDescription", ""),
            "acq_nb_frame": p.get("numImages"),
            "acq_volume_per_image_steps": p.get("volumePerImage"),
            "acq_pump_speed": p.get("pumpSpeed"),
            "acq_stabilization_s": p.get("timeToStabilize"),
            "acq_file_format": p.get("fileFormat"),
            "acq_local_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
            "acq_software": "ImSwitch FlowStop",
        }
        try:
            with open(os.path.join(dirPath, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=4)
        except Exception as e:
            self._logger.error(f"Could not write metadata.json: {e}")

    @APIExport(runOnUIThread=True)
    def snapImageFlowCam(self, fileName=None, fileFormat="JPG"):
        """Save the latest camera frame to `fileName` (extension added here)."""
        if not fileName:
            fileName = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
        mFrame = self.detectorFlowCam.getLatestFrame()
        if mFrame is None:
            self._logger.warning("No frame received from the camera.")
            return
        if fileFormat == "TIF":
            tif.imwrite(fileName + ".tif", mFrame, append=False)
        elif fileFormat in ("JPG", "PNG"):
            ext = ".jpg" if fileFormat == "JPG" else ".png"
            if not cv2.imwrite(fileName + ext, mFrame):
                self._logger.warning(f"Frame could not be saved as {ext} "
                                     f"(shape: {getattr(mFrame, 'shape', None)})")
        else:
            self._logger.warning(f"Nothing saved, unknown file format {fileFormat}")

    def __del__(self):
        self.is_measure = False
        self._stopEvent.set()
        if hasattr(super(), '__del__'):
            super().__del__()


class VideoSafe:
    def __init__(self, frame_provider, output_folder, frame_rate=5, max_frames=1000):
        """Continuously writes frames from `frame_provider` into rolling mp4 chunks."""
        self.frame_provider = frame_provider
        self.output_folder = output_folder
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        self.stop_event = Event()
        self.thread = None
        self.video_writer = None
        self.frame_count = 0
        os.makedirs(output_folder, exist_ok=True)

    def _get_video_writer(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(self.output_folder, f"{timestamp}.mp4")
        frame = self.frame_provider()
        height, width = frame.shape[:2]
        # avc1 plays in browsers; mp4v is the fallback when the codec is unavailable
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*codec),
                                     self.frame_rate, (width, height))
            if writer.isOpened():
                return writer
            writer.release()
        return None

    @staticmethod
    def _toBGR(frame):
        frame = cv2.convertScaleAbs(frame)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame

    def _write_video(self):
        self.video_writer = self._get_video_writer()
        while not self.stop_event.is_set() and self.video_writer is not None:
            frame = self.frame_provider()
            if frame is not None:
                self.video_writer.write(self._toBGR(frame))
                self.frame_count += 1
            if self.frame_count >= self.max_frames:
                self.video_writer.release()
                self.video_writer = self._get_video_writer()
                self.frame_count = 0
            self.stop_event.wait(1 / self.frame_rate)

    def start(self):
        if self.thread is None:
            self.stop_event.clear()
            self.thread = Thread(target=self._write_video, daemon=True)
            self.thread.start()

    def stop(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=5)
            self.thread = None
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
