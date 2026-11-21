import json
import os
import shutil
import threading
from datetime import datetime
import time
import cv2
import numpy as np
from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

from imswitch.imcontrol.model.aufgaben.storage import ImageSpec, NgffImage, save_image
from imswitch.imcontrol.model.io.lightsheet_volumes import (
    data_relative, fov_um, list_volumes, pack_preview, resolve_data_path, tile_layout,
    volume_preview,
)



# ============================================================================
# Pydantic models for API
# ============================================================================

class ScanMode(str, Enum):
    """Scan mode enumeration for lightsheet acquisition"""
    CONTINUOUS = "continuous"  # Original fast scan mode - stage moves continuously
    STEP_ACQUIRE = "step_acquire"  # Go-Stop-Acquire mode - stage moves, stops, acquires

class StorageFormat(str, Enum):
    """Storage format options for lightsheet data"""
    TIFF = "tiff"
    OME_ZARR = "ome_zarr"
    BOTH = "both"

class LightsheetScanParameters(BaseModel):
    """Parameters for lightsheet scan configuration"""
    minPos: float = Field(-500, description="Minimum position for scan axis")
    maxPos: float = Field(500, description="Maximum position for scan axis")
    stepSize: float = Field(10, description="Step size for step-acquire mode (µm)")
    speed: float = Field(1000, description="Speed for continuous mode")
    axis: str = Field("A", description="Scan axis (A, X, Y, Z)")
    illuSource: str = Field("", description="Illumination source name")
    illuValue: float = Field(512, description="Illumination intensity value")
    scanMode: ScanMode = Field(ScanMode.CONTINUOUS, description="Scan mode")
    storageFormat: StorageFormat = Field(StorageFormat.OME_ZARR, description="Storage format")
    experimentName: str = Field("lightsheet_scan", description="Experiment name for file naming")
    currentPosition: float = Field(0, description="Current position of the scan axis")
    
    # Tiling parameters
    enableTiling: bool = Field(False, description="Enable XY tiling")
    tilesXPositive: int = Field(0, description="Number of tiles in positive X direction")
    tilesXNegative: int = Field(0, description="Number of tiles in negative X direction")
    tilesYPositive: int = Field(0, description="Number of tiles in positive Y direction")
    tilesYNegative: int = Field(0, description="Number of tiles in negative Y direction")
    tileStepSizeX: float = Field(0, description="Step size in X for tiling (µm)")
    tileStepSizeY: float = Field(0, description="Step size in Y for tiling (µm)")
    tileOverlap: float = Field(0.1, description="Overlap fraction between tiles (0-1)")
    
    # Timelapse parameters
    timepoints: int = Field(1, description="Number of timepoints to acquire")
    timeLapsePeriod: float = Field(60, description="Period between timepoints (seconds)")

class LightsheetScanStatus(BaseModel):
    """Status information for lightsheet scan"""
    isRunning: bool = False
    scanMode: Optional[str] = None
    currentPosition: float = 0
    totalPositions: int = 0
    currentFrame: int = 0
    progress: float = 0.0
    zarrPath: Optional[str] = None
    tiffPath: Optional[str] = None
    errorMessage: Optional[str] = None
    phase: str = "idle"          # idle | acquiring | writing | done
    currentTile: int = 0         # 1-based tile being acquired (step-acquire)
    totalTiles: int = 1
    zStepUm: Optional[float] = None
    pixelSizeUm: Optional[float] = None


class LightsheetController(ImConWidgetController):
    """Linked to LightsheetWidget."""
    sigImageReceived = Signal(np.ndarray)
    sigSliderIlluValueChanged = Signal(float)
    sigScanStatusUpdate = Signal(dict)  # Signal for scan status updates

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        self.lightsheetTask = None
        self.lightsheetStack = np.ones((1,1,1))
        self.mFilePath = None
        self.mZarrPath = None  # Path to OME-Zarr store

        # Scan state
        self._scanStatus = LightsheetScanStatus()
        self._currentScanParams: Optional[LightsheetScanParameters] = None

        # Observation camera streaming (similar to PixelCalibrationController)
        self.observationStreamRunning = False
        self.observationStreamStarted = False
        self.observationStreamQueue = None

        # Try to get observation camera if available
        try:
            # Look for an observation camera (often named differently from main detector)
            allCameras = self._master.detectorsManager.getAllDeviceNames()
            # Try to find observation camera (you may need to adjust this based on your setup)
            self.observationCamera = None
            for camName in allCameras:
                if "observation" in camName.lower() or "overview" in camName.lower():
                    self.observationCamera = self._master.detectorsManager[camName]
                    self._logger.info(f"Found observation camera: {camName}")
                    break
        except Exception as e:
            self._logger.warning(f"No observation camera found: {e}")
            self.observationCamera = None

        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detectorName = allDetectorNames[0]
        self.detector = self._master.detectorsManager[self.detectorName]

        # select lasers and add to gui
        self.lasers = self._master.lasersManager.getAllDeviceNames()
        self.laser = self.lasers[0]
        self.stageName = self._master.positionersManager.getAllDeviceNames()[0]
        self.stages = self._master.positionersManager[self.stageName]
        self.isLightsheetRunning = False

        # connect signals
        self._commChannel.sigStartLightSheet.connect(self.startContinuousScanWithZarr)
        self._commChannel.sigStopLightSheet.connect(self.stopLightsheet)

    @APIExport()
    def setGalvo(self, channel:int=1, frequency:float=10, offset:float=0, amplitude:float=1, clk_div:int=0, phase:int=0, invert:int=1):
        '''Sets the galvo parameters for the lightsheet.'''
        if not self.isLightsheetRunning:
            try:
                self._master.lasersManager[self.laser].setGalvo(channel=channel, frequency=frequency, offset=offset, amplitude=amplitude, clk_div=clk_div, phase=phase, invert=invert)
                self._logger.info(f"Set galvo parameters: channel={channel}, frequency={frequency}, offset={offset}, amplitude={amplitude}, clk_div={clk_div}, phase={phase}, invert={invert}")
            except Exception as e:
                self._logger.error(f"Error setting galvo parameters: {e}")
        else:
            self._logger.warning("Cannot set galvo parameters while lightsheet is running.")


    '''@APIExport()
    def performScanningRecording(self, minPos:int=0, maxPos:int=1000, speed:int=1000, axis:str="A", illusource:int=-1, illuvalue:int=512):
        """
        [DEPRECATED] Use startContinuousScanWithZarr() instead.
        
        This method is kept for backward compatibility but internally redirects 
        to the new startContinuousScanWithZarr() which supports OME-Zarr storage.
        """
        self._logger.warning("performScanningRecording is deprecated. Use startContinuousScanWithZarr instead.")

        # check parameters
        if axis not in ("A", "X", "Y", "Z"):
            axis = "A"
        # use default illumination source if not selected
        if illusource is None or illusource==-1 or illusource not in self._master.lasersManager.getAllDeviceNames():
            illusource = self._master.lasersManager.getAllDeviceNames()[0]

        # Redirect to new method with default TIFF storage for backward compatibility
        return self.startContinuousScanWithZarr(
            minPos=minPos,
            maxPos=maxPos,
            speed=speed,
            axis=axis,
            illuSource=illusource,
            illuValue=illuvalue,
            storageFormat="tiff",  # Use TIFF for backward compatibility
            experimentName="lightsheet_scan_legacy"
        )'''

    @APIExport()
    def returnLastLightsheetStackPath(self) -> str:
        '''Returns the path of the last saved lightsheet stack.'''
        if self.mFilePath is not None:
            return self.mFilePath
        else:
            return "No stack available yet"


    def getSaveFilePath(self, date, filename, extension):
        mFilename =  f"{date}_{filename}.{extension}"
        dirPath  = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), 'LightSheet', date)
        newPath = os.path.join(dirPath,mFilename)

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        return newPath

    @APIExport()
    def getLatestLightsheetStackAsTif(self):
        """
        If there is a lightsheet stack available, return it as a TIFF file for download.
        """
        if self.mFilePath is not None and os.path.exists(self.mFilePath):
            # Return the file as a response for download
            return FileResponse(
                path=self.mFilePath,
                media_type="application/octet-stream",
                filename=os.path.basename(self.mFilePath)
            )
        else:
            # Return an error message if the file is not available
            return {"error": "No lightsheet stack available"}

    @APIExport()
    def getIsLightsheetRunning(self):
        return self.isLightsheetRunning

    @APIExport()
    def stopLightsheet(self):
        """Abort the running scan. The scan thread stops at the next frame, turns the
        illumination off and still writes what was acquired."""
        self.isLightsheetRunning = False
        try:   # a continuous sweep is one long move; stop it instead of waiting for it
            if hasattr(self.stages, "stopAll"):
                self.stages.stopAll()
        except Exception as e:
            self._logger.warning(f"Could not stop the stage: {e}")
        return {"success": True}

    # ------------------------------------------------------------------ helpers

    def _dataDir(self) -> str:
        return str(dirtools.UserFileDirs.getValidatedDataPath())

    def _lightsheetDir(self) -> str:
        return os.path.join(self._dataDir(), "LightSheet")

    def _newScanDir(self) -> str:
        path = os.path.join(self._lightsheetDir(), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path, exist_ok=True)
        return path

    def _resolveIllumination(self, illuSource) -> str:
        """The laser name to use; the first laser when none (or "-1"/a number) is given."""
        names = self._master.lasersManager.getAllDeviceNames()
        return illuSource if illuSource in names else (names[0] if names else "")

    def _setIllumination(self, name: str, value1024: float):
        """Laser ``name`` at ``value1024``/1024 of its range; 0 turns it off."""
        if not name or name not in self._master.lasersManager.getAllDeviceNames():
            return
        laser = self._master.lasersManager[name]
        laser.setValue((value1024 / 1024) * laser.valueRangeMax)
        laser.setEnabled(value1024 > 0)

    def _frameShape(self):
        """(height, width) of the current camera frame."""
        frame = self.detector.getLatestFrame()
        if frame is not None and frame.ndim >= 2:
            return tuple(frame.shape[:2])
        w, h = self.detector.shape[:2]   # DetectorManager.shape is (width, height)
        return (h, w)

    def _stageAffine(self):
        """``(A, calibrated)``: 2x2 µm per px, camera -> stage XY, from the pixel calibration
        of this detector (PixelCalibrationController), else the plain pixel size."""
        try:
            calib = self._master.getController("PixelCalibration").getCalibrationData(
                detectorName=self.detectorName)
            if calib.get("success"):
                A = np.array(calib["affineMatrix"], dtype=float)[:, :2]
                if abs(np.linalg.det(A)) > 1e-12:
                    return A, True
        except Exception as e:
            self._logger.debug(f"No pixel calibration for {self.detectorName}: {e}")
        px = float(self.detector.pixelSizeUm[-1]) or 1.0
        return np.eye(2) * px, False

    def _writeScanRecord(self, path: str, record: dict):
        """``<name>.scan.json`` next to the data: what the tile map and offline stitching need."""
        try:
            with open(path, "w") as fh:
                json.dump(record, fh, indent=1, default=float)
        except Exception as e:
            self._logger.warning(f"Could not write {path}: {e}")

    # ========================================================================
    # New Go-Stop-Acquire Mode with OME-Zarr support
    # ========================================================================

    def _emitScanStatus(self):
        """Emit scan status via socket and signal"""
        status_dict = self._scanStatus.model_dump()
        self.sigScanStatusUpdate.emit(status_dict)

    @APIExport()
    def getScanStatus(self) -> dict:
        """Get current scan status including progress and file paths."""
        return self._scanStatus.model_dump()

    @APIExport()
    def getAvailableScanModes(self) -> list:
        """Get available scan modes."""
        return [mode.value for mode in ScanMode]

    @APIExport()
    def getAvailableStorageFormats(self) -> list:
        """Get available storage formats."""
        formats = [StorageFormat.TIFF.value]
        formats.append(StorageFormat.OME_ZARR.value)
        formats.append(StorageFormat.BOTH.value)
        return formats

    @APIExport()
    def startStepAcquireScan(
        self,
        minPos: float = -500,
        maxPos: float = 500,
        stepSize: float = 10,
        axis: str = "A",
        illuSource: str = "",
        illuValue: float = 512,
        storageFormat: str = "ome_zarr",
        experimentName: str = "lightsheet_scan",
        enableTiling: bool = False,
        tilesXPositive: int = 0,
        tilesXNegative: int = 0,
        tilesYPositive: int = 0,
        tilesYNegative: int = 0,
        tileStepSizeX: float = 0,
        tileStepSizeY: float = 0,
        tileOverlap: float = 0.1,
        timepoints: int = 1,
        timeLapsePeriod: float = 60
    ) -> dict:
        """
        Start a Go-Stop-Acquire scan with configurable storage format, optional tiling, and timelapse.
        
        This mode moves the stage in discrete steps, stops, acquires an image,
        then moves to the next position. Suitable for high-quality Z-stacks.
        Can optionally acquire multiple XY tiles for large area scanning.
        
        Args:
            minPos: Start position in µm
            maxPos: End position in µm
            stepSize: Step size between acquisitions in µm
            axis: Scan axis (A, X, Y, Z)
            illuSource: Illumination source name
            illuValue: Illumination intensity
            storageFormat: Storage format (tiff, ome_zarr, both)
            experimentName: Name for the experiment/files
            enableTiling: Enable XY tiling
            tilesXPositive: Number of tiles in positive X direction
            tilesXNegative: Number of tiles in negative X direction
            tilesYPositive: Number of tiles in positive Y direction
            tilesYNegative: Number of tiles in negative Y direction
            tileStepSizeX: Step size in X for tiling (µm)
            tileStepSizeY: Step size in Y for tiling (µm)
            tileOverlap: Overlap fraction between tiles (0-1)
            timepoints: Number of timepoints to acquire
            timeLapsePeriod: Period between timepoints (seconds)
            
        Returns:
            dict: Status information including file paths
        """
        if self.isLightsheetRunning:
            return {"error": "Scan already running", "success": False}

        # Validate parameters
        if axis not in ("A", "X", "Y", "Z"):
            axis = "A"

        if stepSize <= 0:
            return {"error": "stepSize must be > 0", "success": False}
        illuSource = self._resolveIllumination(illuSource)

        # Parse storage format
        try:
            storage = StorageFormat(storageFormat.lower())
        except ValueError:
            storage = StorageFormat.OME_ZARR

        # get current position from device:
        currentPosition = self.stages.getPosition()[axis]

        # Create scan parameters
        params = LightsheetScanParameters(
            minPos=currentPosition + minPos, # TODO: we should make these parameters relative to the current position
            maxPos=currentPosition + maxPos,# TODO: we should make these parameters relative to the current position
            currentPosition=currentPosition,
            stepSize=stepSize,
            axis=axis,
            illuSource=illuSource,
            illuValue=illuValue,
            scanMode=ScanMode.STEP_ACQUIRE,
            storageFormat=storage,
            experimentName=experimentName,
            enableTiling=enableTiling,
            tilesXPositive=tilesXPositive,
            tilesXNegative=tilesXNegative,
            tilesYPositive=tilesYPositive,
            tilesYNegative=tilesYNegative,
            tileStepSizeX=tileStepSizeX,
            tileStepSizeY=tileStepSizeY,
            tileOverlap=tileOverlap,
            timepoints=timepoints,
            timeLapsePeriod=timeLapsePeriod
        )

        self._currentScanParams = params
        self.isLightsheetRunning = True

        zPositions = np.arange(params.minPos, params.maxPos + stepSize / 2, stepSize)
        totalZSteps = len(zPositions)
        totalXYTiles = ((tilesXPositive + tilesXNegative + 1) * (tilesYPositive + tilesYNegative + 1)
                        if enableTiling else 1)
        totalPositions = totalZSteps * totalXYTiles * timepoints

        self._scanStatus = LightsheetScanStatus(
            isRunning=True,
            scanMode=ScanMode.STEP_ACQUIRE.value,
            currentPosition=minPos,
            totalPositions=totalPositions,
            currentFrame=0,
            progress=0.0,
            phase="acquiring",
            totalTiles=totalXYTiles,
            zStepUm=abs(stepSize),
        )
        self._emitScanStatus()

        # Start scan in background thread
        if self.lightsheetTask is not None:
            try:
                self.lightsheetTask.join(timeout=1)
            except:
                pass

        self.lightsheetTask = threading.Thread(
            target=self._stepAcquireThread,
            args=(params, zPositions)
        )
        self.lightsheetTask.start()

        return {
            "success": True,
            "message": "Step-acquire scan started",
            "totalPositions": totalPositions,
            "totalZSteps": totalZSteps,
            "totalXYTiles": totalXYTiles,
            "timepoints": timepoints,
            "scanMode": ScanMode.STEP_ACQUIRE.value,
            "storageFormat": storage.value
        }

    # ────────────────────────────────────────────────────────────────────────
    # Deterministic frame-acquisition helpers
    # The same pattern used in ExperimentController / AutofocusController:
    # software-trigger mode guarantees every frame was exposed *after* the
    # stage finished moving.  Falls back to freerun polling transparently.
    # ────────────────────────────────────────────────────────────────────────

    def _getExposureTimeSec(self) -> float:
        """Return current detector exposure time in seconds (best effort)."""
        try:
            exp_ms = self.detector.getParameter("exposure")
            if exp_ms is not None and float(exp_ms) > 0:
                return float(exp_ms) / 1000.0
        except Exception:
            pass
        try:
            return self.detector._camera.exposure_time / 1e6
        except Exception:
            return 0.1

    def _beginTriggeredAcquisition(self) -> bool:
        """Switch the detector to software-trigger mode for deterministic post-move grabs.

        Returns True when successful; False when the detector does not support
        it (graceful degradation to freerun fallback).
        """
        if not hasattr(self.detector, "setTriggerSource") or not hasattr(self.detector, "snapSync"):
            self._useTriggeredGrab = False
            return False
        try:
            ok = bool(self.detector.setTriggerSource("software"))
        except Exception as e:
            self._logger.warning(f"Could not enable software trigger: {e}")
            ok = False
        self._useTriggeredGrab = ok
        if ok:
            self._logger.info("LightsheetController: camera in software-trigger mode (deterministic grabs)")
        else:
            self._logger.warning("LightsheetController: software trigger not supported, using freerun fallback")
        return ok

    def _endTriggeredAcquisition(self) -> None:
        """Restore continuous (free-run) acquisition — safe to call unconditionally."""
        if not getattr(self, "_useTriggeredGrab", False):
            return
        self._useTriggeredGrab = False
        try:
            if hasattr(self.detector, "setTriggerSource"):
                self.detector.setTriggerSource("continuous")
                self._logger.info("LightsheetController: camera restored to continuous mode")
        except Exception as e:
            self._logger.warning(f"Could not restore continuous mode: {e}")

    def grabCameraFrame(self, frameSync: int = 2, returnFrameNumber: bool = False):
        """Return a single camera frame guaranteed to be captured after this call.

        Fast path (triggered): fires one software trigger via detector.snapSync()
        and returns exactly the resulting frame.

        Fallback (freerun): flushes ring-buffer, then polls by frame-number until
        the counter advances by frameSync counts.  Timeout is exposure-time-scaled.
        """
        # ── Fast path ────────────────────────────────────────────────────────
        if getattr(self, "_useTriggeredGrab", False) and hasattr(self.detector, "snapSync"):
            mFrame = self.detector.snapSync(timeout=max(2.0, self._getExposureTimeSec() * 4 + 1.0))
            if returnFrameNumber:
                fn = self.detector.getFrameNumber() if hasattr(self.detector, "getFrameNumber") else -1
                return mFrame, fn
            return mFrame

        # ── Fallback path ────────────────────────────────────────────────────
        try:
            if hasattr(self.detector, "flushBuffer"):
                self.detector.flushBuffer()
        except Exception:
            pass

        exposure_s = self._getExposureTimeSec()
        timeoutFrameRequest = max(1.0, (frameSync + 2) * exposure_s + 0.5)
        cTime = time.time()
        lastFrameNumber = -1
        currentFrameNumber = -1
        mFrame = None

        while True:
            mFrame, currentFrameNumber = self.detector.getLatestFrame(returnFrameNumber=True)
            if lastFrameNumber == -1:
                lastFrameNumber = currentFrameNumber
            if time.time() - cTime > timeoutFrameRequest:
                if mFrame is None:
                    mFrame = self.detector.getLatestFrame(returnFrameNumber=False)
                self._logger.warning(
                    f"grabCameraFrame: timed out after {timeoutFrameRequest:.1f}s "
                    f"(exposure={exposure_s*1000:.0f}ms, frameSync={frameSync})"
                )
                break
            if currentFrameNumber <= lastFrameNumber + frameSync:
                time.sleep(0.01)
            else:
                break

        if returnFrameNumber:
            return mFrame, currentFrameNumber
        return mFrame

    def _planTiles(self, params: LightsheetScanParameters, x0: float, y0: float) -> list:
        """XY tile centres (raster, row by row) around (x0, y0)."""
        if not params.enableTiling:
            return [{"x": x0, "y": y0, "ix": 0, "iy": 0}]
        return [{"x": x0 + ix * params.tileStepSizeX, "y": y0 + iy * params.tileStepSizeY,
                 "ix": ix + params.tilesXNegative, "iy": iy + params.tilesYNegative}
                for iy in range(-params.tilesYNegative, params.tilesYPositive + 1)
                for ix in range(-params.tilesXNegative, params.tilesXPositive + 1)]

    def _stepAcquireThread(self, params: LightsheetScanParameters, zPositions: np.ndarray):
        """Go-stop-acquire: timepoints x XY tiles x Z planes into one stitched canvas.

        Tiles are placed on the canvas with the camera->stage affine of the pixel
        calibration (flips, swapped axes and the real overlap are honoured); without a
        calibration the plain pixel size is used. Overlaps: the later tile wins.
        """
        scanDir = self._newScanDir()
        base = os.path.join(scanDir, params.experimentName)
        initialPosition = self.stages.getPosition()
        tiles = self._planTiles(params, initialPosition["X"], initialPosition["Y"])
        h, w = self._frameShape()
        A, calibrated = self._stageAffine()
        pixelSize = float(np.sqrt(abs(np.linalg.det(A))))
        offsets, (canvasH, canvasW) = tile_layout([(t["x"], t["y"]) for t in tiles], A, (h, w))
        for tile, (row, col) in zip(tiles, offsets):
            tile.update(row=row, col=col, done=False)

        wantZarr = params.storageFormat in (StorageFormat.OME_ZARR, StorageFormat.BOTH)
        wantTiff = params.storageFormat in (StorageFormat.TIFF, StorageFormat.BOTH)
        zarrPath = base + ".ome.zarr" if wantZarr else None
        tiffPath = base + ".ome.tif" if wantTiff else None
        self.mZarrPath, self.mFilePath = zarrPath or self.mZarrPath, tiffPath or self.mFilePath
        self._scanStatus.zarrPath, self._scanStatus.tiffPath = zarrPath, tiffPath
        self._scanStatus.pixelSizeUm = pixelSize

        # Stage position of canvas pixel (0, 0): tile 0's centre, back by its offset + half a frame
        r0, c0 = offsets[0]
        ox, oy = np.array([tiles[0]["x"], tiles[0]["y"]]) - A @ np.array([c0 + w / 2, r0 + h / 2])
        fov = fov_um(A, (h, w))
        record = {
            "name": params.experimentName, "scanMode": ScanMode.STEP_ACQUIRE.value,
            "axis": params.axis, "zRangeUm": [float(zPositions[0]), float(zPositions[-1])],
            "zStepUm": abs(params.stepSize), "pixelSizeUm": pixelSize, "fovUm": list(fov),
            "calibrated": calibrated, "affine": A.tolist(), "tileShape": [h, w],
            "canvasShape": [canvasH, canvasW], "timepoints": params.timepoints, "tiles": tiles,
            "files": {k: data_relative(self._dataDir(), v) for k, v in
                      (("zarr", zarrPath), ("tiff", tiffPath)) if v},
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        recordPath = base + ".scan.json"
        self._writeScanRecord(recordPath, record)
        self._logger.info(f"Step-acquire: {len(tiles)} tile(s) x {len(zPositions)} planes x "
                          f"{params.timepoints} t, canvas {canvasW}x{canvasH} px, "
                          f"calibrated={calibrated}")

        canvas = store = None
        if wantZarr or wantTiff:
            store = zarrPath or os.path.join(scanDir, f".{params.experimentName}.canvas.ome.zarr")
            canvas = NgffImage.create(store, ImageSpec(
                shape=(params.timepoints, 1, len(zPositions), canvasH, canvasW),
                dtype="uint16", pixel_size_um=pixelSize, z_step_um=abs(params.stepSize),
                t_step_s=params.timeLapsePeriod, origin_um=(float(ox), float(oy), float(zPositions[0])),
                channel_names=[params.illuSource or "Lightsheet"], name=params.experimentName,
                chunks_yx=(min(h, 1024), min(w, 1024)),
                annotations={"scanAxis": params.axis, "calibrated": calibrated}))

        frameCounter = 0
        totalFrames = len(zPositions) * len(tiles) * params.timepoints
        try:
            self._setIllumination(params.illuSource, params.illuValue)
            self.detector.startAcquisition()
            # Deterministic grabs: every frame is triggered after the stage settled
            # (falls back to freerun polling when the camera cannot be triggered).
            self._beginTriggeredAcquisition()

            for t_idx in range(params.timepoints):
                if not self.isLightsheetRunning:
                    break
                for tile_idx, tile in enumerate(tiles):
                    if not self.isLightsheetRunning:
                        break
                    self._scanStatus.currentTile = tile_idx + 1
                    self.stages.move(value=tile["x"], axis="X", is_absolute=True, is_blocking=True)
                    self.stages.move(value=tile["y"], axis="Y", is_absolute=True, is_blocking=True)
                    time.sleep(0.1)  # settle
                    for z_idx, zPos in enumerate(zPositions):
                        if not self.isLightsheetRunning:
                            break
                        self.stages.move(value=zPos, axis=params.axis, is_absolute=True, is_blocking=True)
                        time.sleep(0.05)
                        frame = self.grabCameraFrame()
                        if frame is not None and canvas is not None:
                            plane = frame[:, :, 0] if frame.ndim == 3 else frame
                            plane = plane[:h, :w].astype(np.uint16)
                            canvas.write(plane, t_idx, 0, z_idx, tile["row"], tile["col"])
                        frameCounter += 1
                        self._scanStatus.currentFrame = frameCounter
                        self._scanStatus.currentPosition = float(zPos)
                        self._scanStatus.progress = frameCounter / totalFrames * 100
                        self._emitScanStatus()
                    else:
                        tile["done"] = True
                        self._writeScanRecord(recordPath, record)

                if t_idx < params.timepoints - 1 and self.isLightsheetRunning:
                    self._logger.info(f"Waiting {params.timeLapsePeriod}s before the next timepoint")
                    deadline = time.time() + params.timeLapsePeriod
                    while self.isLightsheetRunning and time.time() < deadline:
                        time.sleep(0.2)

            self._logger.info(f"Step-acquire done: {frameCounter} frames")
        except Exception as e:
            self._logger.error(f"Error during step-acquire scan: {e}", exc_info=True)
            self._scanStatus.errorMessage = str(e)
        finally:
            self._endTriggeredAcquisition()
            self._setIllumination(params.illuSource, 0)
            try:
                self.stages.move(value=initialPosition["X"], axis="X", is_absolute=True, is_blocking=True)
                self.stages.move(value=initialPosition["Y"], axis="Y", is_absolute=True, is_blocking=True)
                self.stages.move(value=initialPosition[params.axis], axis=params.axis,
                                 is_absolute=True, is_blocking=False)
            except Exception as e:
                self._logger.warning(f"Could not return to the start position: {e}")
            # What was acquired is written even after an abort or an error
            if canvas is not None:
                self._scanStatus.phase = "writing"
                self._emitScanStatus()
                try:
                    canvas.finalize(min_size=64, max_levels=4)
                    if tiffPath and frameCounter:
                        canvas.export_ome_tiff(tiffPath)
                except Exception as e:
                    self._logger.error(f"Writing the canvas failed: {e}", exc_info=True)
                    self._scanStatus.errorMessage = str(e)
                if not wantZarr:
                    shutil.rmtree(store, ignore_errors=True)
            record.update(frames=frameCounter, finished=datetime.now().isoformat(timespec="seconds"))
            self._writeScanRecord(recordPath, record)
            self.isLightsheetRunning = False
            self._scanStatus.isRunning = False
            self._scanStatus.phase = "done"
            self._scanStatus.progress = 100.0
            self._emitScanStatus()

    @APIExport(runOnUIThread=False)
    def getLatestZarrPath(self) -> dict:
        """The newest lightsheet OME-Zarr (this session's, else the newest on disk).
        ``zarrPath`` is relative to the data folder, as the viewers expect."""
        path = self.mZarrPath if self.mZarrPath and os.path.exists(self.mZarrPath) else None
        if path is None:
            path = next((v["absolutePath"] for v in list_volumes(self._lightsheetDir(), self._dataDir())
                         if v["kind"] == "zarr"), None)
        if path is None:
            return {"zarrPath": None, "exists": False}
        return {"zarrPath": data_relative(self._dataDir(), path), "absolutePath": path, "exists": True}

    @APIExport(runOnUIThread=False)
    def listVolumes(self) -> list:
        """Lightsheet OME-TIFF / OME-Zarr volumes, newest first."""
        return list_volumes(self._lightsheetDir(), self._dataDir())

    @APIExport(runOnUIThread=False)
    def getVolumePreview(self, path: str, maxSize: int = 256, channel: int = 0):
        """A small uint8 copy of a volume for the browser 3D view (read from the coarse
        pyramid level, so it is cheap even for GB stacks).

        ``path`` is relative to the data folder. Body: uint32 LE header length, JSON
        header (shape z/y/x, spacingUm, zStepUm, pixelSizeUm, ...), then the voxels.
        """
        try:
            absPath = resolve_data_path(self._dataDir(), path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"{path} not found")
        maxSize = int(min(max(maxSize, 32), 512))
        try:
            header, volume = volume_preview(absPath, maxSize, maxSize, channel)
        except Exception as e:
            self._logger.error(f"Volume preview of {absPath} failed: {e}", exc_info=True)
            raise HTTPException(status_code=422, detail=f"Cannot read {path}: {e}")
        header.update(path=path, absolutePath=absPath, name=os.path.basename(absPath.rstrip(os.sep)),
                      kind="zarr" if absPath.rstrip(os.sep).endswith(".zarr") else "tiff")
        return Response(content=pack_preview(header, volume), media_type="application/octet-stream")

    @APIExport(runOnUIThread=False)
    def getScannedVolumes(self, limit: int = 30) -> list:
        """Footprints of past scans (``*.scan.json``), newest first, for the tile map."""
        records = []
        root = self._lightsheetDir()
        if os.path.isdir(root):
            for run in sorted(os.listdir(root), reverse=True):
                runDir = os.path.join(root, run)
                if not os.path.isdir(runDir):
                    continue
                for name in sorted(os.listdir(runDir)):
                    if name.endswith(".scan.json"):
                        try:
                            with open(os.path.join(runDir, name)) as fh:
                                records.append({**json.load(fh), "folder": run})
                        except Exception:
                            pass
                if len(records) >= limit:
                    break
        return records[:limit]

    @APIExport(runOnUIThread=False)
    def getObjectiveFOV(self) -> dict:
        """Field of view of one camera frame in stage µm, used for the tile step.

        With a pixel calibration of this detector (Pixel Calibration tab) the FOV, the
        tile placement and the stitched canvas follow its camera->stage affine, so flips,
        a rotated camera and the true overlap are taken into account.
        """
        try:
            A, calibrated = self._stageAffine()
            h, w = self._frameShape()
            fovX, fovY = fov_um(A, (h, w))
            return {
                "fovX": fovX, "fovY": fovY,
                "pixelSize": float(np.sqrt(abs(np.linalg.det(A)))),
                "suggestedOverlap": 0.1,
                "success": True,
                "calibrated": calibrated,
                "detectorName": self.detectorName,
                "affine": A.tolist(),
                "rotationDeg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
                "tileShape": [h, w],
            }
        except Exception as e:
            self._logger.error(f"Error getting the FOV: {e}")
            return {"fovX": 1000, "fovY": 1000, "pixelSize": 1.0, "suggestedOverlap": 0.1,
                    "success": False, "calibrated": False, "error": str(e)}

    @APIExport()
    def startContinuousScanWithZarr(
        self,
        minPos: float = -500,
        maxPos: float = 500,
        speed: float = 1000,
        axis: str = "A",
        illuSource: str = "",
        illuValue: float = 512,
        storageFormat: str = "ome_zarr",
        experimentName: str = "lightsheet_continuous"
    ) -> dict:
        """
        Continuous scan: the stage sweeps from minPos to maxPos (relative to the current
        position) at ``speed`` while every new camera frame is spooled to disk; the stack
        is written as OME-TIFF and/or OME-Zarr afterwards. RAM use stays at one frame.

        Returns:
            dict: Status information
        """
        if self.isLightsheetRunning:
            return {"error": "Scan already running", "success": False}
        if axis not in ("A", "X", "Y", "Z"):
            axis = "A"
        try:
            storage = StorageFormat(storageFormat.lower())
        except ValueError:
            storage = StorageFormat.OME_ZARR
        currentPosition = self.stages.getPosition()[axis]
        params = LightsheetScanParameters(
            minPos=minPos + currentPosition,
            maxPos=maxPos + currentPosition,
            currentPosition=currentPosition,
            speed=speed,
            axis=axis,
            illuSource=self._resolveIllumination(illuSource),
            illuValue=illuValue,
            scanMode=ScanMode.CONTINUOUS,
            storageFormat=storage,
            experimentName=experimentName
        )

        self._currentScanParams = params
        self.isLightsheetRunning = True
        self._scanStatus = LightsheetScanStatus(
            isRunning=True,
            scanMode=ScanMode.CONTINUOUS.value,
            currentPosition=minPos,
            progress=0.0,
            phase="acquiring",
        )
        self._emitScanStatus()

        if self.lightsheetTask is not None:
            self.lightsheetTask.join(timeout=1)
        self.lightsheetTask = threading.Thread(target=self._continuousScanWithZarrThread, args=(params,))
        self.lightsheetTask.start()

        return {
            "success": True,
            "message": "Continuous scan started",
            "scanMode": ScanMode.CONTINUOUS.value,
            "storageFormat": storage.value
        }

    def _latestFrameAndNumber(self):
        try:
            return self.detector.getLatestFrame(returnFrameNumber=True)
        except TypeError:   # detector without frame numbers
            return self.detector.getLatestFrame(), None

    def _continuousScanWithZarrThread(self, params: LightsheetScanParameters):
        """Sweep the scan axis and keep every *new* frame (by frame number), spooled to a
        raw file next to the data; afterwards the spool is memory-mapped and written as
        OME images (the pyramid and the TIFF are streamed, never held in RAM)."""
        self._logger.info(f"Starting continuous scan: {params.minPos} to {params.maxPos}")
        scanDir = self._newScanDir()
        base = os.path.join(scanDir, params.experimentName)
        spool = os.path.join(scanDir, f".{params.experimentName}.spool")
        paths = []
        if params.storageFormat in (StorageFormat.OME_ZARR, StorageFormat.BOTH):
            paths.append(base + ".ome.zarr")
            self.mZarrPath = self._scanStatus.zarrPath = paths[-1]
        if params.storageFormat in (StorageFormat.TIFF, StorageFormat.BOTH):
            paths.append(base + ".ome.tif")
            self.mFilePath = self._scanStatus.tiffPath = paths[-1]

        zs, shape, dtype = [], None, None
        startXY = self.stages.getPosition()
        returnPosition = startXY.get(params.axis, params.currentPosition)
        try:
            self._setIllumination(params.illuSource, params.illuValue)
            self.detector.startAcquisition()
            self.stages.move(value=params.minPos, axis=params.axis, is_absolute=True, is_blocking=True)
            time.sleep(0.5)

            controller = MovementController(self.stages)
            totalDistance = params.maxPos - params.minPos
            controller.move_to_position(totalDistance, params.axis, params.speed, is_absolute=False)
            # Z from the linear time relation: polling the stage would block the serial bus
            startTime = time.time()
            lastNumber, lastFrame = None, None
            with open(spool, "wb") as fh:
                while self.isLightsheetRunning and not controller.is_target_reached():
                    frame, number = self._latestFrameAndNumber()
                    if frame is None or frame.size == 0 or frame is lastFrame or \
                            (number is not None and number == lastNumber):
                        time.sleep(0.002)   # nothing new yet
                        continue
                    lastNumber, lastFrame = number, frame
                    plane = frame[:, :, 0] if frame.ndim == 3 else frame
                    if shape is None:
                        shape, dtype = plane.shape, plane.dtype
                    if plane.shape != shape:
                        continue
                    fh.write(np.ascontiguousarray(plane, dtype=dtype).tobytes())
                    zPos = params.minPos + params.speed * (time.time() - startTime)
                    zs.append(zPos)
                    if len(zs) % 10 == 0:
                        self._scanStatus.currentFrame = len(zs)
                        self._scanStatus.currentPosition = zPos
                        self._scanStatus.progress = min(abs(zPos - params.minPos) / abs(totalDistance) * 100, 100)
                        self._emitScanStatus()
        except Exception as e:
            self._logger.error(f"Error in continuous scan: {e}", exc_info=True)
            self._scanStatus.errorMessage = str(e)
        finally:
            self._setIllumination(params.illuSource, 0)
            try:
                self.stages.move(value=returnPosition, axis=params.axis, is_absolute=True, is_blocking=False)
            except Exception as e:
                self._logger.warning(f"Could not return to the start position: {e}")
            try:
                n = len(zs)
                dz = abs(zs[-1] - zs[0]) / max(1, n - 1) if n > 1 else 1.0
                pixelSize = float(self.detector.pixelSizeUm[-1])
                self._scanStatus.zStepUm, self._scanStatus.pixelSizeUm = dz, pixelSize
                if n and paths:
                    self._scanStatus.phase = "writing"
                    self._emitScanStatus()
                    stack = np.memmap(spool, dtype=dtype, mode="r", shape=(n,) + tuple(shape))
                    for path in paths:
                        save_image(path, stack, "ZYX", pixel_size_um=pixelSize, z_step_um=dz,
                                   origin_um=(startXY.get("X", 0.0), startXY.get("Y", 0.0), min(zs)),
                                   channel_names=[params.illuSource or "Lightsheet"],
                                   annotations={"scanAxis": params.axis, "speed": params.speed})
                        self._logger.info(f"Wrote {n} planes to {path}")
                    del stack
                A, calibrated = self._stageAffine()
                h, w = shape if shape else self._frameShape()
                self._writeScanRecord(base + ".scan.json", {
                    "name": params.experimentName, "scanMode": ScanMode.CONTINUOUS.value,
                    "axis": params.axis, "zRangeUm": [min(zs), max(zs)] if zs else [params.minPos, params.maxPos],
                    "zStepUm": dz, "pixelSizeUm": pixelSize, "fovUm": list(fov_um(A, (h, w))),
                    "calibrated": calibrated, "tileShape": [h, w], "frames": n,
                    "tiles": [{"x": startXY.get("X", 0.0), "y": startXY.get("Y", 0.0), "row": 0, "col": 0, "done": True}],
                    "files": {("zarr" if p.endswith(".zarr") else "tiff"): data_relative(self._dataDir(), p) for p in paths},
                    "finished": datetime.now().isoformat(timespec="seconds"),
                })
            except Exception as e:
                self._logger.error(f"Writing the continuous scan failed: {e}", exc_info=True)
                self._scanStatus.errorMessage = str(e)
            finally:
                try:
                    os.remove(spool)
                except OSError:
                    pass
                self.isLightsheetRunning = False
                self._scanStatus.isRunning = False
                self._scanStatus.phase = "done"
                self._scanStatus.progress = 100.0
                self._emitScanStatus()

    # ========================================================================
    # Observation Camera Streaming (for sample positioning visualization)
    # ========================================================================

    @APIExport()
    def stopObservationStream(self):
        """Stop the observation camera MJPEG stream."""
        self.observationStreamRunning = False
        self.observationStreamStarted = False
        self.observationStreamQueue = None

    def startObservationStream(self):
        """
        Background thread that converts observation camera frames to JPEG and queues them.
        """

        if self.observationCamera is None:
            self._logger.error("Observation camera not available")
            return

        # Wait for first valid frame (up to 2s); fall back to black frame
        deadline = time.time() + 2.0
        output_frame = None
        while self.observationStreamRunning and output_frame is None and time.time() < deadline:
            try:
                output_frame = self.observationCamera.getLatestFrame()
            except Exception:
                output_frame = None
            if output_frame is None:
                time.sleep(0.05)

        if output_frame is None:
            # Default black frame if nothing available (grayscale)
            output_frame = np.zeros((480, 640), dtype=np.uint8)

        # Adaptive resize: Keep frames below 640x480
        try:
            if output_frame.shape[0] > 640 or output_frame.shape[1] > 480:
                everyNthsPixel = int(
                    np.min(
                        [
                            max(1, output_frame.shape[0] // 480),
                            max(1, output_frame.shape[1] // 640),
                        ]
                    )
                )
            else:
                everyNthsPixel = 1
        except Exception:
            everyNthsPixel = 1

        try:
            while self.observationStreamRunning:
                output_frame = self.observationCamera.getLatestFrame()
                if output_frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    # Downsample if needed
                    output_frame = output_frame[::everyNthsPixel, ::everyNthsPixel]
                except Exception:
                    output_frame = np.zeros((480, 640), dtype=np.uint8)

                # Convert grayscale to BGR if needed (for consistent processing)
                if len(output_frame.shape) == 2:
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

                # Ensure uint8 image for JPEG; normalize if needed
                if output_frame.dtype != np.uint8:
                    try:
                        vmin = float(np.min(output_frame))
                        vmax = float(np.max(output_frame))
                        if vmax > vmin:
                            output_frame = (
                                (output_frame - vmin) / (vmax - vmin) * 255.0
                            ).astype(np.uint8)
                        else:
                            output_frame = np.zeros_like(output_frame, dtype=np.uint8)
                    except Exception:
                        output_frame = np.zeros_like(output_frame, dtype=np.uint8)

                # JPEG compression
                quality = 90  # Quality level (0-100)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                flag, encodedImage = cv2.imencode(".jpg", output_frame, encode_params)
                if not flag:
                    continue

                # Put raw JPEG bytes into queue; avoid blocking forever if queue is full
                try:
                    self.observationStreamQueue.put(encodedImage.tobytes(), timeout=0.5)
                except Exception:
                    # Drop frame if queue is full or unavailable
                    pass

                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            self._logger.error(f"Observation stream error: {e}", exc_info=True)
            self.observationStreamRunning = False

    def observationStreamer(self):
        """
        Generator that yields JPEG frames from the queue.
        Starts the background streaming thread if not already running.
        """
        import queue

        # Start the streaming worker thread once and create a thread-safe queue
        if not self.observationStreamStarted:
            import threading

            self.observationStreamQueue = queue.Queue(maxsize=10)
            self.observationStreamRunning = True
            self.observationStreamStarted = True
            t = threading.Thread(target=self.startObservationStream, daemon=True)
            t.start()

        try:
            while self.observationStreamRunning:
                try:
                    # Use timeout to allow graceful shutdown
                    jpeg_bytes = self.observationStreamQueue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Build proper MJPEG part with Content-Length for better client compatibility
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                )
                yield header + jpeg_bytes + b"\r\n"
        except GeneratorExit:
            self._logger.debug("Observation stream connection closed by client.")
            self.stopObservationStream()

    @APIExport(runOnUIThread=False)
    def observationStream(self, startStream: bool = True):
        """
        Get MJPEG stream from observation camera for sample positioning.
        
        Args:
            startStream: Whether to start the stream (default True)
            
        Returns:
            StreamingResponse with multipart/x-mixed-replace for MJPEG stream
        """
        if not startStream:
            self.stopObservationStream()
            return {"status": "success", "message": "stream stopped"}

        if self.observationCamera is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=409, detail="Observation camera not available")

        from fastapi.responses import StreamingResponse
        headers = {
            # Disable buffering and caching to reduce latency
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }

        return StreamingResponse(
            self.observationStreamer(),
            media_type="multipart/x-mixed-replace;boundary=frame",
            headers=headers,
        )

    # ========================================================================
    # Camera Settings API (Exposure, Gain for live view)
    # ========================================================================

    # ========================================================================
    # Virtual lightsheet (VirtualMicroscope with "imagePath": "lightsheet")
    # ========================================================================

    def _virtualLightsheet(self):
        try:
            vm = self._master.rs232sManager["VirtualMicroscope"]
            return vm if vm.getLightsheetOptics() is not None else None
        except Exception:
            return None

    @APIExport(runOnUIThread=False)
    def getVirtualOptics(self) -> dict:
        """Optics of the simulated lightsheet (sheet waist, Rayleigh range, resolution...);
        ``available`` is False on a real microscope."""
        vm = self._virtualLightsheet()
        return {"available": False} if vm is None else {"available": True, **vm.getLightsheetOptics()}

    @APIExport(runOnUIThread=False)
    def setVirtualIlluminationNA(self, na: float) -> dict:
        """Illumination NA of the simulated sheet: thinner sheet, shorter usable field."""
        vm = self._virtualLightsheet()
        if vm is None:
            return {"available": False, "error": "No virtual lightsheet in this setup"}
        return {"available": True, **vm.setLightsheetIlluminationNA(float(na))}

    def _getCameraParameter(self, name: str, key: str) -> dict:
        p = self.detector.parameters.get(name)
        if p is None:
            return {key: None, "success": False, "error": f"{name} parameter not available"}
        return {key: p.value, "min": getattr(p, "valueMin", None), "max": getattr(p, "valueMax", None),
                "units": getattr(p, "valueUnits", ""), "success": True}

    def _setCameraParameter(self, name: str, value: float, key: str) -> dict:
        """Through the detector manager, so the camera itself changes (not only the cache)."""
        try:
            self.detector.setParameter(name, float(value))
            return self._getCameraParameter(name, key)
        except Exception as e:
            self._logger.error(f"Error setting {name}: {e}")
            return {"success": False, "error": str(e)}

    @APIExport()
    def getCameraExposureTime(self) -> dict:
        """Exposure time (ms) of the lightsheet camera with its valid range."""
        return self._getCameraParameter("exposure", "exposureTime")

    @APIExport()
    def setCameraExposureTime(self, exposureTime: float) -> dict:
        """Set the exposure time (ms); returns the value the camera accepted."""
        return self._setCameraParameter("exposure", exposureTime, "exposureTime")

    @APIExport()
    def getCameraGain(self) -> dict:
        """Gain of the lightsheet camera with its valid range."""
        return self._getCameraParameter("gain", "gain")

    @APIExport()
    def setCameraGain(self, gain: float) -> dict:
        """Set the gain; returns the value the camera accepted."""
        return self._setCameraParameter("gain", gain, "gain")


class MovementController:
    def __init__(self, stages):
        self.stages = stages
        self.target_reached = False
        self.target_position = None
        self.axis = None

    def move_to_position(self, minPos, axis, speed, is_absolute):
        self.target_position = minPos
        self.speed = speed
        self.is_absolute = is_absolute
        self.axis = axis
        thread = threading.Thread(target=self._move)
        thread.start()

    def _move(self):
        self.target_reached = False
        self.stages.move(value=self.target_position, axis=self.axis, speed=self.speed, is_absolute=self.is_absolute, is_blocking=True)
        self.target_reached = True

    def is_target_reached(self):
        return self.target_reached

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
