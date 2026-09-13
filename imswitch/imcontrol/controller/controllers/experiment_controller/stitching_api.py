"""Ashlar stitching, run on demand (never automatically).

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
import os
import subprocess
import sys
import threading
import time
from typing import Dict, Any

from imswitch.imcommon.model import APIExport


class AshlarMixin:
    """Ashlar stitching, run on demand (never automatically)."""

    @APIExport(requestType="GET")
    def runAshlarStitching(
        self,
        pixelSize: float = 1.0,
        maximumShift: float = 50.0,
        alignChannel: int = 0,
        experimentDir: str = "",
    ) -> dict:
        """
        Schedule ashlarUC2 stitching for the last (or specified) experiment
        directory and return immediately.

        The actual stitching runs in a background thread identical to the
        overview-scan async pattern.  Poll ``getOverviewAsyncStatus`` for
        completion; the result dict will contain ``outputDir`` on success.

        Parameters
        ----------
        pixelSize     : physical pixel size in µm/pixel
        maximumShift  : max per-tile corrective shift in µm (ashlar -m)
        alignChannel  : channel index used for alignment (ashlar -c)
        experimentDir : absolute path to a specific experiment directory;
                        uses the most recent experiment when empty
        """
        # Resolve which experiment directory to stitch before spawning the thread
        target_dir = experimentDir.strip() if experimentDir else ""
        if not target_dir:
            target_dir = getattr(self, "_last_experiment_dir", "")
        if not target_dir or not os.path.isdir(target_dir):
            return {
                "started": False,
                "error": (
                    f"Experiment directory not found: '{target_dir}'. "
                    "Run an experiment first or provide experimentDir."
                ),
            }

        if not self._tryStartOverviewAsync("ashlar_stitching"):
            return {
                "started": False,
                "error": "Another background task is still running",
                "status": self._overview_async_status,
            }

        # Resolve pixel size: when the UI still shows the default (1.0) read the
        # calibrated value that PixelCalibrationController pushed into the detector
        # from PixelCalibration.affineCalibrations in the ImSwitchConfig file.
        resolved_pixel_size = pixelSize
        if pixelSize == 1.0:
            try:
                det_px = self.mDetector.pixelSizeUm  # [Z, Y, X]
                cal_px = float(det_px[1]) if len(det_px) > 1 else 1.0
                if cal_px > 0 and cal_px != 1.0:
                    resolved_pixel_size = cal_px
                    self._logger.info(
                        f"Ashlar: using calibrated pixel size from detector: {cal_px:.4f} µm/px"
                    )
            except Exception as exc:
                self._logger.warning(f"Could not read detector pixel size: {exc}")

        thread = threading.Thread(
            target=self._runAshlarStitchingWorker,
            args=(target_dir, resolved_pixel_size, maximumShift, alignChannel),
            daemon=True,
            name="runAshlarStitching",
        )
        with self._overview_async_lock:
            self._overview_async_thread = thread
        thread.start()
        return {"started": True, "task": "ashlar_stitching", "experimentDir": target_dir}


    def _runAshlarStitchingWorker(
        self,
        target_dir: str,
        pixelSize: float,
        maximumShift: float,
        alignChannel: int,
    ) -> None:
        """
        Worker thread body for ashlar stitching.
        Calls _finishOverviewAsync so getOverviewAsyncStatus reflects completion.
        """
        self._logger.info(f"Ashlar stitching started for: {target_dir}")

        # Locate convert_experiment_tiffs.py relative to this controller file.
        # imswitch/imcontrol/controller/controllers/ → root → scripts/
        script = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..", "scripts",
                "convert_experiment_tiffs.py",
            )
        )
        if not os.path.isfile(script):
            self._logger.error(f"Stitching script not found: {script}")
            self._finishOverviewAsync(
                error=f"convert_experiment_tiffs.py not found at: {script}"
            )
            return

        cmd = [
            sys.executable, script,
            target_dir,
            "--mode", "ashlar",
            "--maximum-shift", str(maximumShift),
            "--align-channel", str(alignChannel),
            "--pixel-size", str(pixelSize),
        ]
        self._logger.info(f"Ashlar command: {' '.join(cmd)}")

        deadline = time.monotonic() + 7200  # 2-hour hard timeout

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with self._overview_async_lock:
                self._ashlar_proc = proc

            stderr_lines: list[str] = []

            def _stream():
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    self._logger.info(f"[ashlar] {line}")
                    stderr_lines.append(line)
                    with self._overview_async_lock:
                        if self._overview_async_status.get("running"):
                            self._overview_async_status["message"] = line

            reader_thread = threading.Thread(target=_stream, daemon=True)
            reader_thread.start()

            # Wait for process with a per-second timeout check so we honour
            # the 2-hour deadline even if Popen.wait() doesn't time out itself.
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    proc.kill()
                    reader_thread.join(timeout=5)
                    self._logger.error("Ashlar stitching timed out after 2 hours")
                    self._finishOverviewAsync(error="Stitching timed out after 2 hours")
                    return
                time.sleep(1)

            reader_thread.join(timeout=10)
            with self._overview_async_lock:
                self._ashlar_proc = None
            rc = proc.returncode

            if rc == 0:
                output_dir = os.path.join(target_dir, "converted", "ashlar")
                output_files = []
                if os.path.isdir(output_dir):
                    output_files = [
                        f for f in os.listdir(output_dir)
                        if f.endswith(".tif") or f.endswith(".tiff")
                    ]
                if output_files:
                    self._logger.info(f"Ashlar stitching completed → {output_dir} ({len(output_files)} file(s))")
                    self._finishOverviewAsync(result={
                        "status": "done",
                        "outputDir": output_dir,
                        "files": output_files,
                        "message": stderr_lines[-1] if stderr_lines else "",
                    })
                else:
                    tail = "\n".join(stderr_lines[-40:]) or "Script exited 0 but no stitched files were written."
                    self._logger.error(f"Ashlar stitching produced no output files. Script output:\n{tail}")
                    self._finishOverviewAsync(error=tail)
            else:
                tail = "\n".join(stderr_lines[-20:]) or f"Script exited with code {rc}"
                self._logger.error(f"Ashlar stitching failed (rc={rc}): {tail}")
                self._finishOverviewAsync(error=tail)

        except Exception as exc:
            self._logger.error(f"Ashlar stitching worker exception: {exc}", exc_info=True)
            self._finishOverviewAsync(error=str(exc))

    @APIExport(requestType="GET")
    def stopAshlarStitching(self) -> dict:
        """Kill the running Ashlar stitching subprocess, if any."""
        with self._overview_async_lock:
            proc = self._ashlar_proc
        if proc is None or proc.poll() is not None:
            return {"stopped": False, "message": "No stitching process is running"}
        proc.kill()
        self._logger.info("Ashlar stitching process killed by user request")
        self._finishOverviewAsync(error="Stitching stopped by user")
        return {"stopped": True}

    # ------------------------------------------------------------------
    # Known calibration reference points per layout
    # ------------------------------------------------------------------
    # Hard-coded reference centres (in micrometres) for known calibration
    # targets. ``StageOffsetCalibrationTab`` shows these as the "expected"
    # position; the deviation from the brightest spot detected by the raster
    # scan is what gets persisted as the stage offset.
    _KNOWN_CALIBRATION_POINTS: Dict[str, Dict[str, Any]] = {
        "Heidstar 4x Histosample": {
            "x": 20000.0,
            "y": 40000.0,
            "description": "Centre of the openUC2 calibration pinhole on the Heidstar 4x slide carrier (slot 1).",
        },
        "openUC2 96-Well Calibration Chart": {
            "x": 14380.0,
            "y": 11240.0,
            "description": "Centre well A1 of the openUC2 96-well calibration chart (slot 1).",
        },
    }
