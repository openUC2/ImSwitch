from datetime import datetime
import time
from fastapi import HTTPException
import numpy as np
from typing import List, Dict, Any

import os
import threading

from imswitch.imcommon.framework import Signal
from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext, WorkflowsManager
from imswitch.imcontrol.model.managers.MDASequenceManager import MDASequenceManager
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController

from imswitch.imcontrol.model.io.frame_save_worker import FrameSaveWorker
from imswitch.imcontrol.controller.controllers.experiment_controller import ExperimentPerformanceMode, ExperimentNormalMode, ExecutionContext, Experiment, ParameterValue
from imswitch.imcontrol.controller.controllers.experiment_controller.scan_plan import resolve_channels, z_offsets, writer_flags, build_scan_regions, regions_to_areas
from imswitch.imcontrol.model.focus_map import FocusMapManager


from imswitch.imcontrol.controller.controllers.experiment_controller.labware_api import LabwareMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.stitching_api import AshlarMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.fast_stage_scan import FastStageScanMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.mda_api import MDAMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.focus_map_api import FocusMapMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.overview_api import OverviewRegistrationMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.hardware_api import HardwareMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.autofocus_api import AutofocusMixin
from imswitch.imcontrol.controller.controllers.experiment_controller.frame_saving import FrameSavingMixin


class ExperimentController(LabwareMixin, AshlarMixin, FastStageScanMixin, MDAMixin, FocusMapMixin,
                           OverviewRegistrationMixin, HardwareMixin, AutofocusMixin, FrameSavingMixin,
                           ImConWidgetController):
    """Linked to ExperimentWidget."""

    sigExperimentWorkflowUpdate = Signal()
    sigExperimentImageUpdate = Signal(str, np.ndarray, bool, list, bool)  # (detectorName, image, init, scale, isCurrentDetector)
    sigUpdateOMEZarrStore = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # initialize variables
        self.tWait = 0.1
        self.workflow_manager = WorkflowsManager()
        self.mda_manager = MDASequenceManager()

        # Background frame-save worker. ``save_frame_ome`` and the
        # individual-TIFF path it dispatches to used to call into
        # ``ome_writer.write_frame`` / ``tifffile.imwrite`` *on the
        # workflow thread*, which blocked the next stage step for
        # 20–100 ms per frame at 9 MP (more for stitched/individual
        # TIFFs, much more for a slow SD/USB target). Now we just
        # queue the write here and the worker thread does the I/O.
        # The queue is bounded so a slow disk back-pressures the
        # acquisition loop instead of silently consuming RAM.
        self._frame_saver = FrameSaveWorker(
            name="ExperimentFrameSaver",
            maxsize=32,
        )

        # set default values
        self.SPEED_Y = self.SPEED_Y_default = 20000
        self.SPEED_X = self.SPEED_X_default = 20000
        self.SPEED_Z = self.SPEED_Z_default = 10000
        self.ACCELERATION = self.ACCELERATION_default = 1000000
        self.ACCELERATION_Z = self.ACCELERATION_Z_default = 1000000
        # Global Z focus offset updated at runtime by autofocus during an
        # experiment. Capture Z-moves add this on top of their base Z so the
        # measured focus is actually applied (see move_stage_z / autofocus).
        # Reset to 0.0 at the start of every experiment.
        # Autofocus Z offset per region_id. A single shared scalar let one bad
        # point defocus every region after it.
        self._experiment_af_offsets = {}
        self._af_successes = 0
        self._af_failures = 0

        self._init_hardware()
        # stop if some external signal (e.g. memory full is triggered)
        self._commChannel.sigExperimentStop.connect(self.stopExperiment)

        # where to dump the TIFFs ----------------------------------------------
        save_dir = dirtools.UserFileDirs.getValidatedDataPath()
        self.save_dir  = os.path.join(save_dir, "ExperimentController")
        # ensure all subfolders are generated:
        os.makedirs(self.save_dir, exist_ok=True)

        # writer thread control -------------------------------------------------
        self._writer_thread   = None
        self._writer_thread_ome = None
        self._current_ome_writer = None  # For normal mode OME writing
        self._stop_writer_evt = threading.Event()

        # fast stage scanning parameters ----------------------------------------
        self.fastStageScanIsRunning = False

        # OME writer configuration -----------------------------------------------
        self._ome_write_tiff = False
        self._ome_write_zarr = True
        self._ome_write_stitched_tiff = False
        self._ome_write_individual_tiffs = False
        self._ome_write_single_tiff = False

        # Initialize experiment execution modes
        self.performance_mode = ExperimentPerformanceMode(self)
        self.normal_mode = ExperimentNormalMode(self)
        # True while startWellplateExperiment is setting a run up — see the
        # guard there for why the workflow status alone is not enough.
        self._experiment_starting = False
        # Scan areas of the most recent run. Read by the focus-map endpoints,
        # which can be called before any run has happened.
        self._last_scan_areas = None
        self._focus_map_active = False

        # Initialize focus map manager
        self.focus_map_manager = FocusMapManager(logger=self._logger)

        self._init_overview()

        # Initialize Opentrons-style labware manager. A single bad labware file
        # must never block controller startup, so we swallow exceptions.
        try:
            from imswitch.imcontrol.model.labware import LabwareManager as _LabwareManager
            self.labware_manager = _LabwareManager()
        except Exception as exc:
            self._logger.warning(f"LabwareManager init failed: {exc}")
            self.labware_manager = None

        # Initialize omero  parameters  # TODO: Maybe not needed!
        self.omero_url = self._master.experimentManager.omeroServerUrl
        self.omero_username = self._master.experimentManager.omeroUsername
        self.omero_password = self._master.experimentManager.omeroPassword
        self.omero_port = self._master.experimentManager.omeroPort

    @APIExport(requestType="GET")
    def getExperimentStatus(self):
        """Get the current status of running experiments."""
        # Check workflow manager status (normal mode)
        if self.ExperimentParams.performanceMode:
            # Check performance mode status
            workflow_status = self.performance_mode.get_scan_status()
        else:
            # Check normal mode status
            workflow_status = self.workflow_manager.get_status()

        # Autofocus outcome: a run can "complete" with every autofocus failed.
        return {
            **workflow_status,
            "autofocus": {
                "successes": self._af_successes,
                "failures": self._af_failures,
                "offsets_um": dict(self._experiment_af_offsets),
            },
        }

    @APIExport(requestType="POST")
    def startWellplateExperiment(self, mExperiment: Experiment):
        """Start a wellplate experiment, refusing a second concurrent start.

        Setup takes minutes with focus mapping on, and the workflow guard
        inside only reports "running" once the workflow starts — after the
        focus-map phase. This flag covers that window; the workflow /
        performance-scan status guards take over from there.
        """
        if self._experiment_starting:
            raise HTTPException(
                status_code=400, detail="An experiment is already being started."
            )
        self._experiment_starting = True
        try:
            return self._start_wellplate_experiment(mExperiment)
        finally:
            self._experiment_starting = False

    def _start_wellplate_experiment(self, mExperiment: Experiment):
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")
        exp_name = mExperiment.name
        p = mExperiment.parameterValue

        # 1. Plan — pure, see scan_plan.py.
        plan = resolve_channels(p)
        for note in plan.notes:
            self._logger.info(note)
        if not plan.passthrough:
            p.illuIntensities = plan.intensities  # model props (n_active_channels) count synthetic channels too
        p.performanceMode = plan.performance_mode
        self.ExperimentParams.performanceMode = plan.performance_mode
        snake_tiles, region_meta = build_scan_regions(mExperiment, self.mStage.getPosition, self._logger)
        self._last_scan_areas = regions_to_areas(snake_tiles, region_meta)
        z_positions = z_offsets(p)
        for flag, value in writer_flags(p, snake_tiles).items():
            setattr(self, f"_ome_{flag}", value)

        # 2. Side effects, in the order the hardware needs them.
        self._apply_run_settings(p)
        self._illuminationIntensities = plan.intensities  # autofocus_software looks up its channel here
        self._illuminationSources = plan.sources
        self.set_led_status("rainbow")
        if not self.mDetector._running:
            self.mDetector.startAcquisition()

        self._focus_map_fit_by_region = True
        self._focus_map_settle_ms = 0
        self._focus_map_active = False  # a map left over from an earlier run is never re-applied
        focusMapConfig = mExperiment.focusMap
        if focusMapConfig is not None and focusMapConfig.enabled:
            self._logger.info("Focus mapping enabled – computing Z surface per scan group")
            self._focus_map_fit_by_region = focusMapConfig.fit_by_region
            self._focus_map_settle_ms = focusMapConfig.settle_ms
            self._focus_map_active = bool(getattr(focusMapConfig, "apply_during_scan", True))
            self._run_focus_map_phase(self._last_scan_areas, focusMapConfig,
                                      af_kwargs=self._focus_map_af_kwargs(p))
        self._switch_off_all_illumination()

        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirPath = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), 'ExperimentController', timeStamp)
        os.makedirs(dirPath, exist_ok=True)
        self._last_experiment_dir = dirPath
        self._start_i2c_logging(dirPath)

        pos = self.mStage.getPosition()
        self._initial_experiment_position = {"X": pos.get("X", 0), "Y": pos.get("Y", 0), "Z": pos.get("Z", 0)}
        ctx = ExecutionContext(
            experiment=mExperiment, snake_tiles=snake_tiles, z_positions=z_positions,
            illumination_sources=plan.sources, illumination_intensities=plan.intensities,
            exposures=plan.exposures, gains=plan.gains,
            exp_name=exp_name, dir_path=dirPath, file_name=f"{timeStamp}_{exp_name}",
            initial_xyz=self._initial_experiment_position,
            keep_illumination_on=plan.keep_illumination_on, is_rgb=self.isRGB,
            illumination_kinds=plan.kinds, illumination_params=plan.params, region_meta=region_meta,
        )

        # 3. Execute.
        if plan.performance_mode and self.performance_mode.is_hardware_capable():
            self.performance_mode.execute_experiment(ctx=ctx)
            return {"status": "running", "mode": "performance"}
        self._run_normal_mode(ctx)
        return {"status": "running"}

    def _apply_run_settings(self, p: ParameterValue) -> None:
        """Per-run stage/exposure knobs from the request; <= 0 means the __init__ default."""
        self.SPEED_X = p.speed if p.speed > 0 else self.SPEED_X_default
        self.SPEED_Y = p.speed if p.speed > 0 else self.SPEED_Y_default
        self.SPEED_Z = p.z_speed if p.z_speed > 0 else self.SPEED_Z_default
        self.ACCELERATION = p.acceleration if p.acceleration > 0 else self.ACCELERATION_default
        self.ACCELERATION_Z = p.z_acceleration if p.z_acceleration > 0 else self.ACCELERATION_Z_default
        # Every exposure wrapped in bus-power off → settle → grab → on → settle (see acquire_frame).
        self._busPowerDarkness = bool(p.busPowerDarkness)
        self._busPowerSettle = float(p.busPowerSettleTime or 0.0)
        if self._busPowerDarkness:
            self._logger.info(f"CAN-bus power darkness ENABLED (settle={self._busPowerSettle:.1f}s each side of every exposure)")
        self._return_to_origin = bool(p.returnToOrigin)
        # Autofocus state never leaks between runs.
        self._experiment_af_offsets = {}
        self._af_successes = 0
        self._af_failures = 0

    def _run_normal_mode(self, ctx: ExecutionContext) -> None:
        """Build the workflow for every timepoint and start it on the workflow manager."""
        workflow_steps, file_writers = [], []
        for t in range(ctx.n_times):
            ctx.timepoint_index = t
            result = self.normal_mode.execute_experiment(ctx=ctx)
            workflow_steps.extend(result["workflow_steps"])
            file_writers.extend(result["file_writers"])

        self._step_timings = []  # per-step timings for the post-run violin plot

        def sendProgress(payload):
            try:
                if isinstance(payload, dict) and payload.get("status") == "completed":
                    self._step_timings.append({
                        "name": payload.get("name", "step"),
                        **{k: float(payload.get(k, 0.0) or 0.0) for k in ("pre_time", "main_time", "post_time")},
                    })
            except Exception:
                pass
            self.sigExperimentWorkflowUpdate.emit(payload)

        wf = Workflow(workflow_steps, self.workflow_manager)
        context = WorkflowContext()
        context.set_metadata("experimentName", ctx.exp_name)
        context.set_metadata("nTimes", ctx.n_times)
        context.set_metadata("tPeriod", ctx.t_period)
        context.set_metadata("experiment_start_time", time.time())
        context.set_metadata("timepoint_times", {})
        if file_writers:
            context.set_object("file_writers", file_writers)
        context.on("progress", sendProgress)
        context.on("rgb_stack", sendProgress)

        # Deterministic triggered acquisition for every acquire_frame(); falls back
        # to frame-number polling when the detector has no software trigger.
        self._beginTriggeredAcquisition()
        self.workflow_manager.start_workflow(wf, context)

        # Cleanup once the workflow thread ends, also on stop/abort. Writers are
        # held here, not read back from the context: the manager clears
        # context.objects before the thread exits.
        wf_thread = self.workflow_manager.current_thread
        writers = list(file_writers)

        def _cleanup():
            if wf_thread is not None:
                wf_thread.join()
            for w in writers:  # a stop skips the finalize steps; finalize() is idempotent
                try:
                    w.finalize()
                except Exception as err:
                    self._logger.error(f"Post-workflow writer cleanup failed: {err}")
            self._endTriggeredAcquisition()
            self._export_step_timings_plot(ctx.dir_path)

        threading.Thread(target=_cleanup, daemon=True, name="TriggeredAcqCleanup").start()
        # Ashlar is never started automatically; use runAshlarStitching().


    def computeScanRanges(self, snake_tiles):
        """Compute scan ranges - delegated to base class method."""
        return self.performance_mode.compute_scan_ranges(snake_tiles)


    ########################################
    # Hardware-related functions
    ########################################

    @staticmethod
    def _categorize_step(name: str) -> str:
        """Group a workflow step name into a coarse category for the timing
        violin plot."""
        n = (name or "").strip().lower()
        if "autofocus" in n or "focus map" in n:
            return "Autofocus"
        if "acquire" in n or "snap" in n:
            return "Acquire"
        if n.startswith("move") or "z offset" in n or "→ z" in n or "z →" in n:
            return "Move"
        if "illumination" in n or "laser" in n or "led" in n:
            return "Illumination"
        if "save" in n or "finaliz" in n or "writer" in n:
            return "Save"
        if "wait" in n or "timepoint" in n:
            return "Wait"
        return "Other"

    def _export_step_timings_plot(self, out_dir: str) -> None:
        """Render a violin plot of per-step wall-clock durations (grouped by
        category) and save it as ``step_timings_violin.png`` in ``out_dir``.

        Best-effort: any failure (no data, matplotlib missing, unwritable dir)
        is logged and swallowed so it never affects the experiment outcome.
        """
        timings = list(getattr(self, "_step_timings", []) or [])
        if not timings or not out_dir:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Bucket total step durations (pre+main+post) by category.
            buckets: Dict[str, List[float]] = {}
            for t in timings:
                total = (t.get("pre_time", 0.0) + t.get("main_time", 0.0)
                         + t.get("post_time", 0.0))
                cat = self._categorize_step(t.get("name", ""))
                buckets.setdefault(cat, []).append(total)

            # Keep a stable, meaningful category order.
            order = ["Move", "Autofocus", "Illumination", "Acquire", "Save",
                     "Wait", "Other"]
            labels = [c for c in order if c in buckets]
            data = [buckets[c] for c in labels]
            if not data:
                return

            fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 5))
            positions = list(range(1, len(labels) + 1))
            # Violins need >1 point; fall back to a scatter for single-sample
            # categories so they still render.
            multi_idx = [i for i, d in enumerate(data) if len(d) > 1]
            if multi_idx:
                ax.violinplot([data[i] for i in multi_idx],
                              positions=[positions[i] for i in multi_idx],
                              showmeans=True, showextrema=True)
            for i, d in enumerate(data):
                jitter = np.random.uniform(-0.08, 0.08, size=len(d))
                ax.scatter(np.full(len(d), positions[i]) + jitter, d,
                           s=12, alpha=0.5, color="tab:blue")

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [f"{c}\n(n={len(buckets[c])})" for c in labels])
            ax.set_ylabel("Step duration (s)")
            total_steps = len(timings)
            total_time = sum(
                t.get("pre_time", 0.0) + t.get("main_time", 0.0)
                + t.get("post_time", 0.0) for t in timings)
            ax.set_title(
                f"Per-step timing — {total_steps} steps, "
                f"{total_time:.1f}s total")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "step_timings_violin.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            self._logger.info(f"Saved step-timing violin plot → {out_path}")
        except Exception as e:
            self._logger.warning(f"Could not export step-timing plot: {e}",
                                 exc_info=True)

    def dummy_main_func(self):
        self._logger.debug("Dummy main function called")
        return True

    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        time.sleep(seconds)

    def wait_for_next_timepoint(self, timepoint: int, t_period: float, context: WorkflowContext, metadata: Dict[str, Any]):
        """
        Wait for the proper time interval between timepoints, accounting for measurement time.

        Args:
            timepoint: Current timepoint index
            t_period: Target period between timepoints in seconds
            context: WorkflowContext containing timing information
            metadata: Metadata dictionary
        """
        current_time = time.time()
        experiment_start_time = context.get_metadata("experiment_start_time", current_time)
        timepoint_times = context.get_metadata("timepoint_times", {})

        # Calculate expected time for this timepoint
        expected_time = experiment_start_time + (timepoint + 1) * t_period

        # Store timing information for this timepoint
        timepoint_times[str(timepoint)] = current_time
        context.set_metadata("timepoint_times", timepoint_times)

        # Calculate how long to wait
        wait_time = max(0, expected_time - current_time)

        if wait_time > 0:
            self._logger.info(f"Waiting {wait_time:.2f}s for next timepoint (timepoint {timepoint})")
            time.sleep(wait_time)
        else:
            self._logger.warning(f"Timepoint {timepoint} is running {abs(wait_time):.2f}s behind schedule")
            # Small delay to prevent issues
            time.sleep(0.01)

    @APIExport()
    def pauseWorkflow(self):
        """Pause the workflow. Only works in normal mode."""
        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check if performance mode is running
        performance_status = self.performance_mode.get_scan_status()
        if performance_status["running"]:
            return {"status": "error", "message": "Cannot pause experiment in performance mode"}

        if workflow_status == "running":
            return self.workflow_manager.pause_workflow()
        else:
            return {"status": "error", "message": f"Cannot pause in current state: {workflow_status}"}

    @APIExport()
    def resumeExperiment(self):
        """Resume the experiment. Only works in normal mode."""
        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check if performance mode is running
        performance_status = self.performance_mode.get_scan_status()
        if performance_status["running"]:
            return {"status": "error", "message": "Cannot resume experiment in performance mode"}

        if workflow_status == "paused":
            return self.workflow_manager.resume_workflow()
        else:
            return {"status": "error", "message": f"Cannot resume in current state: {workflow_status}"}

    @APIExport()
    def stopExperiment(self):
        """Stop the experiment. Works for both normal and performance modes."""
        # Abort any in-progress focus map computation first so
        # the synchronous _run_focus_map_phase loop exits early.
        try:
            self.focus_map_manager.request_abort()
        except Exception:
            pass

        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check performance mode status
        performance_status = self.performance_mode.get_scan_status()

        results = {}

        # Stop workflow if running
        if workflow_status in ["running", "paused", "stopping"]:
            results["workflow"] = self.workflow_manager.stop_workflow()

        # Stop performance mode if running
        if performance_status["running"]:
            results["performance"] = self.performance_mode.stop_scan()

        # Return to initial XYZ position if stored
        try:
            if hasattr(self, "_initial_experiment_position") and self._initial_experiment_position:
                pos = self._initial_experiment_position
                self._logger.info(
                    "Returning to initial position: X=%.2f, Y=%.2f, Z=%.2f",
                    pos["X"], pos["Y"], pos["Z"],
                )
                self.move_stage_xy(pos["X"], pos["Y"], relative=False)
                self.move_stage_z(pos["Z"], relative=False)
                self._initial_experiment_position = None
        except Exception as e:
            self._logger.warning("Failed to return to initial position: %s", e)

        # Set LED status to idle
        self.set_led_status("idle")

        # Drain any frame writes that are still queued on the
        # background saver — the workflow has stopped producing new
        # frames, so this should normally be a brief wait. Bounded so
        # that a stuck disk doesn't hang stopExperiment forever; if
        # the timeout fires, the saver thread keeps running in the
        # background and the writes still eventually land on disk.
        try:
            if getattr(self, "_frame_saver", None) is not None:
                pending = self._frame_saver.pending()
                if pending:
                    self._logger.info(
                        f"Draining {pending} pending frame write(s)…"
                    )
                drained = self._frame_saver.flush(timeout=30.0)
                if not drained:
                    self._logger.warning(
                        f"Frame saver still has "
                        f"{self._frame_saver.pending()} pending write(s) "
                        f"after 30 s — they will complete in the background."
                    )
        except Exception as e:
            self._logger.debug(f"Frame saver flush failed: {e}")

        # If nothing was running, return appropriate message
        if not results:
            return "No experiments are currently running"

        return results


    @APIExport()
    def forceStopExperiment(self):
        """Force stop the experiment. Works for both normal and performance modes."""
        results = {}

        # Force stop workflow
        try:
            self.workflow_manager.stop_workflow()
            del self.workflow_manager
            self.workflow_manager = WorkflowsManager()
            results["workflow"] = {"status": "force_stopped", "message": "Workflow force stopped"}
        except Exception as e:
            results["workflow"] = {"status": "error", "message": f"Error force stopping workflow: {e}"}
            self.set_led_status("error")

        # Force stop performance mode
        try:
            results["performance"] = self.performance_mode.force_stop_scan()
        except Exception as e:
            results["performance"] = {"status": "error", "message": f"Error force stopping performance mode: {e}"}
            self.set_led_status("error")

        # Set LED status to idle if no errors
        if all(r.get("status") != "error" for r in results.values()):
            self.set_led_status("idle")

        return results


    """Couples a 2‑D stage scan with external‑trigger camera acquisition.

    • Puts the connected ``CameraHIK`` into *external* trigger mode
      (one exposure per TTL rising edge on LINE0).
    • Runs ``positioner.start_stage_scanning``.
    • Pops every frame straight from the camera ring‑buffer and writes it to
      disk as ``000123.tif`` (frame‑id used as filename).

    Assumes the micro‑controller (or the positioner itself) raises a TTL pulse
    **after** arriving at each grid co‑ordinate.
    """
