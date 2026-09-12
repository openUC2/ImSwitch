"""Hardware-triggered fast stage scan.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
from datetime import datetime
import time
from typing import List

import os
import threading

from imswitch.imcommon.model import dirtools, APIExport

from imswitch.imcontrol.model.io import OMEFileStorePaths


class FastStageScanMixin:
    """Hardware-triggered fast stage scan."""

    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisition(self,
                      xstart:float=0, xstep:float=500, nx:int=10,
                      ystart:float=0, ystep:float=500, ny:int=10,
                      zstart:float=0, zstep:float=0, nz:int=1,
                      tsettle:float=90, tExposure:float=50,
                      illumination:List[int]=None, led:float=None,
                      tPeriod:int=1, nTimes:int=1,
                      isSnakeScan:bool=True):
        """Full workflow: arm camera ➔ launch writer ➔ execute scan.

        Args:
            xstart: Starting X position
            xstep: Step size in X direction
            nx: Number of steps in X
            ystart: Starting Y position
            ystep: Step size in Y direction
            ny: Number of steps in Y
            zstart: Starting Z position (for Z-stacking)
            zstep: Step size in Z direction (0 = no Z-stacking)
            nz: Number of Z planes (1 = single plane)
            tsettle: Settle time after movement (ms)
            tExposure: Exposure time (ms)
            illumination: List of illumination channel intensities (e.g., [100, 50, 0, 75, 0])
            led: LED intensity (0-255)
            tPeriod: Period between time points (s)
            nTimes: Number of time points
            isSnakeScan: If True, apply snake (serpentine) scan pattern; if False, use raster
        """
        self.fastStageScanIsRunning = True
        self._stop() # ensure all prior runs are stopped
        self.move_stage_xy(posX=xstart, posY=ystart, relative=False)

        # Turn off all illumination channels before starting scan
        self._switch_off_all_illumination()

        # compute the metadata for the stage scan (e.g. x/y coordinates and illumination channels)
        # stage will start at xstart, ystart and move in steps of xstep, ystep in snake scan logic

        # Ensure illumination is a list
        if illumination is None:
            illumination = []

        # Build illumination dict for metadata (maintain backward compatibility)
        illum_dict = {}
        for i, val in enumerate(illumination[:5]):  # Take up to 5 channels
            illum_dict[f"illumination{i}"] = val
        illum_dict["led"] = led

        # Count how many illumination entries are valid (not None and > 0)
        nIlluminations = sum(val is not None and val > 0 for val in illumination) + (1 if led and led > 0 else 0)
        nScan = max(nIlluminations, 1)
        total_frames = nx * ny * nz * nScan
        self._logger.info(f"Stage-scan: {nx}×{ny}×{nz} ({total_frames} frames)")

        def addDataPoint(metadataList, x, y, z, illuminationChannel, illuminationValue, runningNumber):
            """Helper function to add metadata for each position."""
            metadataList.append({
                "x": x,
                "y": y,
                "z": z,
                "illuminationChannel": illuminationChannel,
                "illuminationValue": illuminationValue,
                "runningNumber": runningNumber
            })
            return metadataList
        # This corresponds to the metadataList in the UC2-ESP firmware e.g. here: https://github.com/youseetoo/uc2-esp32/blob/9addafaca538186e642e97f50c248df661a74637/main/src/motor/StageScan.cpp#L602
        metadataList = []
        runningNumber = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    z = zstart + iz * zstep
                    x = xstart + ix * xstep
                    y = ystart + iy * ystep
                    # Snake pattern: reverse X direction on odd rows
                    if isSnakeScan and iy % 2 == 1:
                        x = xstart + (nx - 1 - ix) * xstep

                    # If there's at least one valid illumination or LED set, take only one image as "default"
                    if nIlluminations == 0:
                        runningNumber += 1
                        addDataPoint(metadataList, x, y, z, "default", -1, runningNumber)
                    else:
                        # Otherwise take an image for each illumination channel > 0
                        for channel, value in illum_dict.items():
                            if value is not None and value > 0:
                                runningNumber += 1
                                addDataPoint(metadataList, x, y, z, channel, value, runningNumber)
        # 2. start writer thread ----------------------------------------------
        nLastTime = time.time()
        for iTime in range(nTimes):
            saveOMEZarr = True
            nTimePoints = nTimes
            nZPlanes = nz

            if saveOMEZarr:
                # ------------------------------------------------------------------+
                # 2. open OME-Zarr canvas                                           |
                # ──────────────────────────────────────────────────────────────────+
                timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.mFilePath = os.path.join(self.save_dir,  f"{timeStamp}_FastStageScan")
                # create directory if it does not exist and file paths
                omezarr_store = OMEFileStorePaths(self.mFilePath)
                data_path = dirtools.UserFileDirs.getValidatedDataPath()
                self.setOmeZarrUrl(self.mFilePath.split(data_path)[-1]+".ome.zarr")
                self._writer_thread_ome = threading.Thread(
                    target=self._writer_loop_ome, args=(omezarr_store, total_frames, metadataList, xstart, ystart, xstep, ystep, nx, ny, 0, nTimePoints, nZPlanes, nIlluminations),
                    daemon=True)
                self._stop_writer_evt.clear()
                self._writer_thread_ome.start()
            else:
                # Non-performance mode: also use _writer_loop_ome but configure for TIFF stitching
                self._stop_writer_evt.clear()
                self._writer_thread_ome = threading.Thread(
                    target=self._writer_loop_ome,
                    args=(omezarr_store, total_frames, metadataList, xstart, ystart, xstep, ystep, nx, ny, 0.2, True, True, False, nTimePoints, nZPlanes, nIlluminations),  # is_tiff=True, write_stitched_tiff=True, is_performance_mode=False
                    daemon=True
                )
                self._writer_thread_ome.start()

            # Pad illumination list to 5 channels if needed
            illumination_padded = (illumination + [0] * 5)[:5] if illumination else [0, 0, 0, 0, 0]

            # 3. execute stage scan (blocks until finished) ------------------------
            self.fastStageScanIsRunning = True  # Set flag to indicate scan is running
            self.mStage.start_stage_scanning(
                xstart=0, xstep=xstep, nx=nx, # we choose xstart/ystart = 0 since this means we start from here in the positive direction with nsteps
                ystart=0, ystep=ystep, ny=ny,
                zstart=0, zstep=zstep, nz=nz,  # Z-stacking parameters
                tsettle=tsettle, tExposure=tExposure,
                illumination=tuple(illumination_padded), led=led,
            )
            # Wait for time period or until scan is stopped
            while nLastTime + tPeriod < time.time() and self.fastStageScanIsRunning:
                time.sleep(0.1) # TODO: This fails / breaks immediately if there is no timealpse as the start_stage_scanning runs in the background and does not give a signal if it'S done
        return self.getOmeZarrUrl()  # return relative path to the data directory

    def _stop(self):
        """Abort the acquisition gracefully."""
        self._stop_writer_evt.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
        if self._writer_thread_ome is not None:
            self._writer_thread_ome.join(timeout=2)
        self.mDetector.stopAcquisition()

    @APIExport(runOnUIThread=False)
    def stopFastStageScanAcquisition(self):
        """Stop the stage scan acquisition and writer thread."""
        self.mStage.stop_stage_scanning()
        self.fastStageScanIsRunning = False
        self._logger.info("Stopping stage scan acquisition...")
        self._stop()
        self._logger.info("Stage scan acquisition stopped.")

    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisitionFilePath(self) -> str:
        """Returns the file path of the last saved fast stage scan."""
        if hasattr(self, 'fastStageScanFilePath') and self.fastStageScanFilePath is not None:
            return self.fastStageScanFilePath
        else:
            return "No fast stage scan available yet"

    # MDA (Multi-Dimensional Acquisition) Methods using useq-schema
