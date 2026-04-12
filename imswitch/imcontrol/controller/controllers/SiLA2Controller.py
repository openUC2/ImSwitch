"""
SiLA2Controller for OpenUC2 ImSwitch.

Creates concrete SiLA2 feature implementations that delegate to ImSwitch
managers and controllers, registers them with the SiLA2Manager, and starts
the SiLA2 server.

Pattern follows ArkitektController: the controller reads managers from
``self._master`` and wires them to the SiLA2 feature implementations.
"""

import asyncio
import base64
import io
import json
import time
from typing import Optional

import numpy as np

from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcontrol.model.sila2_features import (
    StageControlFeature,
    ImagingControlFeature,
    ExperimentControlFeature,
)

# ---------------------------------------------------------------------------
# Concrete feature implementations
# ---------------------------------------------------------------------------


class _StageControlImpl(StageControlFeature):
    """Concrete SiLA2 stage control backed by ImSwitch PositionersManager."""

    def __init__(self, master, logger):
        super().__init__()
        self._master = master
        self._logger = logger

    def _get_positioner(self, name: Optional[str] = None):
        """Return the positioner manager instance."""
        names = self._master.positionersManager.getAllDeviceNames()
        if not names:
            return None
        positioner_name = name or names[0]
        return self._master.positionersManager[positioner_name]

    async def get_stage_position(self) -> str:
        positioner = self._get_positioner()
        if positioner is None:
            return "0,0,0"
        pos = positioner.getPosition()
        x = pos.get("X", 0)
        y = pos.get("Y", 0)
        z = pos.get("Z", 0)
        return f"{x},{y},{z}"

    async def move_stage_to(
        self,
        x_um: float,
        y_um: float,
        z_um: float = 0.0,
        speed: float = 10000.0,
        is_blocking: bool = True,
    ) -> bool:
        positioner = self._get_positioner()
        if positioner is None:
            self._logger.error("No positioner available")
            return False
        try:
            positioner.move(
                value=(x_um, y_um),
                axis="XY",
                is_absolute=True,
                is_blocking=is_blocking,
                speed=(speed, speed),
            )
            if z_um != 0.0:
                positioner.move(
                    value=z_um,
                    axis="Z",
                    is_absolute=True,
                    is_blocking=is_blocking,
                    speed=speed,
                )
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 move_stage_to failed: {e}")
            return False

    async def move_stage_relative(
        self,
        dx_um: float = 0.0,
        dy_um: float = 0.0,
        dz_um: float = 0.0,
        speed: float = 10000.0,
        is_blocking: bool = True,
    ) -> bool:
        positioner = self._get_positioner()
        if positioner is None:
            self._logger.error("No positioner available")
            return False
        try:
            if dx_um != 0.0 or dy_um != 0.0:
                positioner.move(
                    value=(dx_um, dy_um),
                    axis="XY",
                    is_absolute=False,
                    is_blocking=is_blocking,
                    speed=(speed, speed),
                )
            if dz_um != 0.0:
                positioner.move(
                    value=dz_um,
                    axis="Z",
                    is_absolute=False,
                    is_blocking=is_blocking,
                    speed=speed,
                )
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 move_stage_relative failed: {e}")
            return False

    async def home_stage(self, is_blocking: bool = True) -> bool:
        positioner = self._get_positioner()
        if positioner is None:
            return False
        try:
            if hasattr(positioner, "moveToSampleLoadingPosition"):
                positioner.moveToSampleLoadingPosition(is_blocking=is_blocking)
            elif hasattr(positioner, "home"):
                positioner.home(is_blocking=is_blocking)
            else:
                self._logger.warning("Positioner has no home/loading position method")
                return False
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 home_stage failed: {e}")
            return False


class _ImagingControlImpl(ImagingControlFeature):
    """Concrete SiLA2 imaging control backed by ImSwitch DetectorsManager / LasersManager."""

    def __init__(self, master, logger):
        super().__init__()
        self._master = master
        self._logger = logger

    def _get_detector(self, name: str = ""):
        names = self._master.detectorsManager.getAllDeviceNames()
        if not names:
            return None
        detector_name = name if name and name in names else names[0]
        return self._master.detectorsManager[detector_name]

    async def get_available_detectors(self) -> str:
        names = self._master.detectorsManager.getAllDeviceNames()
        return ",".join(names)

    async def get_available_illumination_sources(self) -> str:
        names = list(self._master.lasersManager.getAllDeviceNames())
        return ",".join(names)

    async def snap_image(
        self,
        detector_name: str = "",
        exposure_time_ms: float = -1.0,
        gain: float = -1.0,
    ) -> str:
        detector = self._get_detector(detector_name)
        if detector is None:
            self._logger.error("No detector available for snap_image")
            return ""
        try:
            # Optionally set exposure / gain
            if exposure_time_ms > 0:
                detector.setParameter("exposure", exposure_time_ms)
            if gain >= 0:
                detector.setParameter("gain", gain)

            # Acquire frame
            frame = detector.getLatestFrame()
            if frame is None:
                return ""

            arr = np.array(frame)

            # Encode as PNG → base64
            from PIL import Image as PILImage

            if arr.ndim == 2:
                img = PILImage.fromarray(arr)
            else:
                img = PILImage.fromarray(arr)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return b64
        except Exception as e:
            self._logger.error(f"SiLA2 snap_image failed: {e}")
            return ""

    async def set_illumination(
        self,
        channel_name: str,
        intensity: float,
        enabled: bool = True,
    ) -> bool:
        try:
            laser_manager = self._master.lasersManager
            all_names = laser_manager.getAllDeviceNames()
            if channel_name not in all_names:
                self._logger.error(
                    f"Illumination channel '{channel_name}' not found. "
                    f"Available: {all_names}"
                )
                return False
            laser = laser_manager[channel_name]
            laser.setValue(intensity)
            laser.setEnabled(1 if enabled else 0)
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 set_illumination failed: {e}")
            return False

    async def set_exposure_time(
        self,
        exposure_time_ms: float,
        detector_name: str = "",
    ) -> bool:
        detector = self._get_detector(detector_name)
        if detector is None:
            return False
        try:
            detector.setParameter("exposure", exposure_time_ms)
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 set_exposure_time failed: {e}")
            return False


class _ExperimentControlImpl(ExperimentControlFeature):
    """Concrete SiLA2 experiment control backed by ImSwitch ExperimentController."""

    def __init__(self, master, logger):
        super().__init__()
        self._master = master
        self._logger = logger
        self._experiment_controller = None

    def _get_experiment_controller(self):
        """Lazily resolve the ExperimentController (registered after __init__)."""
        if self._experiment_controller is None:
            self._experiment_controller = self._master.getController("Experiment")
        return self._experiment_controller

    async def start_experiment(self, experiment_json: str) -> str:
        ec = self._get_experiment_controller()
        if ec is None:
            self._logger.error("ExperimentController not available – cannot start experiment")
            return '{"error": "ExperimentController not available"}'

        try:
            # Parse the incoming JSON into the Experiment Pydantic model used
            # by ExperimentController.startWellplateExperiment()
            from imswitch.imcontrol.controller.controllers.ExperimentController import (
                Experiment,
            )

            experiment = Experiment.model_validate_json(experiment_json)
            self._logger.info(f"SiLA2: starting experiment '{experiment.name}'")

            # Delegate to the ExperimentController (runs synchronously in its
            # own thread internally)
            ec.startWellplateExperiment(experiment)

            return json.dumps({"status": "started", "name": experiment.name})
        except Exception as e:
            self._logger.error(f"SiLA2 start_experiment failed: {e}")
            return json.dumps({"error": str(e)})

    async def get_experiment_status(self) -> str:
        ec = self._get_experiment_controller()
        if ec is None:
            return json.dumps({"status": "unknown", "detail": "ExperimentController unavailable"})
        try:
            status = ec.getExperimentStatus()
            return json.dumps(status) if isinstance(status, dict) else str(status)
        except Exception as e:
            self._logger.error(f"SiLA2 get_experiment_status failed: {e}")
            return json.dumps({"error": str(e)})

    async def pause_experiment(self) -> bool:
        ec = self._get_experiment_controller()
        if ec is None:
            return False
        try:
            ec.pauseExperiment()
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 pause_experiment failed: {e}")
            return False

    async def resume_experiment(self) -> bool:
        ec = self._get_experiment_controller()
        if ec is None:
            return False
        try:
            ec.resumeExperiment()
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 resume_experiment failed: {e}")
            return False

    async def stop_experiment(self) -> bool:
        ec = self._get_experiment_controller()
        if ec is None:
            return False
        try:
            ec.stopExperiment()
            return True
        except Exception as e:
            self._logger.error(f"SiLA2 stop_experiment failed: {e}")
            return False

    async def get_experiment_schema(self) -> str:
        try:
            from imswitch.imcontrol.controller.controllers.ExperimentController import (
                Experiment,
            )

            return Experiment.model_json_schema()
        except Exception as e:
            self._logger.error(f"SiLA2 get_experiment_schema failed: {e}")
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SiLA2Controller(ImConWidgetController):
    """
    Controller for the SiLA2 integration in ImSwitch.

    Creates concrete SiLA2 feature implementations that delegate to ImSwitch
    hardware managers, registers them with the SiLA2Manager, and starts the
    SiLA2 server.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._logger.debug("Initializing SiLA2Controller")

        # Guard: check that the SiLA2Manager is available and enabled
        if not getattr(self._master, "sila2Manager", None):
            self._logger.warning("SiLA2Manager unavailable; controller disabled.")
            return
        if not self._master.sila2Manager.is_enabled():
            self._logger.warning("SiLA2 not enabled; controller disabled.")
            return

        manager = self._master.sila2Manager

        # Create and register feature implementations
        stage_feature = _StageControlImpl(self._master, self._logger)
        imaging_feature = _ImagingControlImpl(self._master, self._logger)
        experiment_feature = _ExperimentControlImpl(self._master, self._logger)

        manager.register_feature(stage_feature)
        manager.register_feature(imaging_feature)
        manager.register_feature(experiment_feature)

        # Start the SiLA2 server
        manager.start_server()
        self._logger.info("SiLA2Controller initialized – server starting")

    # ------------------------------------------------------------------
    # API-exported helpers (accessible via ImSwitch REST API as well)
    # ------------------------------------------------------------------

    @APIExport(requestType="GET")
    def getSiLA2Status(self) -> dict:
        """Return current SiLA2 server status."""
        mgr = getattr(self._master, "sila2Manager", None)
        if mgr is None:
            return {"enabled": False, "running": False}
        return {
            "enabled": mgr.is_enabled(),
            "running": mgr.is_running(),
            "config": mgr.get_config(),
        }
