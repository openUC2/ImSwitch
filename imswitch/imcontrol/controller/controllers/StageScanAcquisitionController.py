import os
import time
import threading
import collections
import tifffile as tif
from pathlib import Path
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController


class StageScanAcquisitionController(ImConWidgetController):
    """Couples a 2‑D stage scan with external‑trigger camera acquisition.

    • Puts the connected ``CameraHIK`` into *external* trigger mode
      (one exposure per TTL rising edge on LINE0).
    • Runs ``positioner.start_stage_scanning``.
    • Pops every frame straight from the camera ring‑buffer and writes it to
      disk as ``000123.tif`` (frame‑id used as filename).

    Assumes the micro‑controller (or the positioner itself) raises a TTL pulse
    **after** arriving at each grid co‑ordinate.
    """

    def __init__(self, *args, save_dir: str | os.PathLike | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # shortcuts to hardware -------------------------------------------------
        self.camera    = self._master.detectorsManager.getAllDeviceNames()[0]
        self.camera    = self._master.detectorsManager[self.camera]   # CameraHIK
        pos_name       = self._master.positionersManager.getAllDeviceNames()[0]
        self.positioner= self._master.positionersManager[pos_name]

        # where to dump the TIFFs ----------------------------------------------
        self.save_dir  = Path(save_dir or Path.home() / "scan")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # writer thread control -------------------------------------------------
        self._writer_thread   = None
        self._stop_writer_evt = threading.Event()

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------
    @APIExport(runOnUIThread=False)
    def startStageScanAcquisition(self,
                      xstart:float=0, xstep:float=500, nx:int=10,
                      ystart:float=0, ystep:float=500, ny:int=10,
                      tsettle:float=10, tExposure:float=50,
                      illumination0:int=None, illumination1:int=None,
                      illumination2:int=None, illumination3:int=None, led:float=None):
        """Full workflow: arm camera ➔ launch writer ➔ execute scan."""
        illumination = (illumination0, illumination1, illumination2, illumination3)
        total_frames = nx * ny
        self._logger.info(f"Stage‑scan: {nx}×{ny} ({total_frames} frames)")

        # 1. prepare camera ----------------------------------------------------
        self.camera.stopAcquisition()
        #self.camera.NBuffer        = total_frames + 32   # head‑room
        #self.camera.frame_buffer   = collections.deque(maxlen=self.camera.NBuffer)
        #self.camera.frameid_buffer = collections.deque(maxlen=self.camera.NBuffer)
        self.camera.setTriggerSource("External trigger")
        self.camera.flushBuffers()
        self.camera.startAcquisition()

        # 2. start writer thread ----------------------------------------------
        self._stop_writer_evt.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(total_frames,),
            daemon=True,
        )
        self._writer_thread.start()

        # 3. execute stage scan (blocks until finished) ------------------------
        self.positioner.start_stage_scanning(
            xstart=xstart, xstep=xstep, nx=nx,
            ystart=ystart, ystep=ystep, ny=ny,
            tsettle=tsettle, tExposure=tExposure,
            illumination=illumination, led=led,
        )

        
    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _writer_loop(self, n_expected: int):
        """
        Bulk-writer that uses the (frames, ids) tuple returned by camera.getChunk().
        The call is non-blocking; it returns empty arrays until new frames arrive.
        """
        saved = 0
        while saved < n_expected or not self._stop_writer_evt.is_set():

            frames, ids = self.camera.getChunk()        # ← empties camera buffer

            if frames.size == 0:                        # nothing new yet
                time.sleep(0.005)
                continue

            for frame, fid in zip(frames, ids):
                tif.imwrite(self.save_dir / f"{fid:06d}.tif", frame)
                saved += 1
                self._logger.debug(f"saved {saved}/{n_expected}")

        self._logger.info(f"Writer thread finished ({saved} images).")
        # 4. wait for writer to finish then stop camera ------------------------
        self.camera.stop_live()
        self._logger.info("Grid‑scan completed and all images saved.")

        # 5. clean up and bring camera back to normal mode -----------------
        self.camera.stopAcquisition()
        self.camera.setTriggerSource("Continuous")
        self.camera.flushBuffers()
        self.camera.startAcquisition()
        self._logger.info("Camera reset to continuous mode.")

    # -------------------------------------------------------------------------
    # tear‑down helpers
    # -------------------------------------------------------------------------
    def stop(self):
        """Abort the acquisition gracefully."""
        self._stop_writer_evt.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
        self.camera.stop_live()

    @APIExport(runOnUIThread=False)
    def stopStageScanAcquisition(self):
        """Stop the stage scan acquisition and writer thread."""
        self._logger.info("Stopping stage scan acquisition...")
        self.stop()
        self._logger.info("Stage scan acquisition stopped.")